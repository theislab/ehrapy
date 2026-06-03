from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from ehrdata import EHRData


_MTTKRP_EINSUM = ["ijk,jr,kr->ir", "ijk,ir,kr->jr", "ijk,ir,jr->kr"]


def _nonneg_cp(
    tensor: np.ndarray,
    rank: int,
    n_iter_max: int = 300,
    tol: float = 1e-8,
    random_state: int = 0,
    sparsity: float = 0.0,
    orthogonality: float = 0.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    r"""Non-negative CP decomposition via Multiplicative Updates (Lee & Seung).

    Uses three optimisations over a naive implementation:

    * **einsum MTTKRP** — the matricised-tensor-times-Khatri–Rao-product is computed via ``np.einsum`` without ever forming the Khatri–Rao matrix.
    * **Gram-matrix denominator** — ``kr.T @ kr`` is replaced by the Hadamard product of the other factors' Gram matrices (O(R²) instead of O(JK·R²)).
    * **Algebraic error** — the reconstruction error is computed from Gram matrices and the cached MTTKRP, avoiding the full-tensor reconstruction.

    Optional regularisation acts on the **variable factor** :math:`B` (mode 1) only, the factor whose
    interpretation as a per-program disease signature the constraints are meant to sharpen:

    * **sparsity** (L1) adds a uniform offset to the update denominator, multiplicatively shrinking and
      driving small loadings toward zero so each disease loads on *few* programs. The offset is sized
      relative to the mean explained magnitude (``sparsity × mean(denominator)``) so the same value behaves
      consistently regardless of factor scale or tensor dimensions.
    * **orthogonality** penalises off-diagonal entries of :math:`B^\\top B`, adding
      :math:`\\lambda_\\perp\\, B (\\mathbf{1}\\mathbf{1}^\\top - I)` to the denominator so two programs
      avoid sharing the same disease — i.e. *fewer features per program*.

    Both penalties have a non-negative gradient in :math:`B`, so they enter the denominator and keep the
    update multiplicative and the factors non-negative.

    Args:
        tensor: Non-negative input tensor of shape ``(I, J, K)``.
        rank: Number of components.
        n_iter_max: Maximum number of iterations.
        tol: Convergence tolerance on the relative change in reconstruction error.
        random_state: Seed for initialisation.
        sparsity: L1 penalty strength :math:`\\lambda_1` on the variable factor. ``0`` disables it.
        orthogonality: Off-diagonal :math:`B^\\top B` penalty strength :math:`\\lambda_\\perp`. ``0`` disables it.

    Returns:
        weights
            Per-component weights of shape ``(rank,)``.
        factors
            List of non-negative factor matrices, one per tensor mode.
    """
    xp = tensor.__array_namespace__()

    rng = np.random.default_rng(random_state)
    eps = 1e-12

    factors = [xp.asarray(np.abs(rng.standard_normal((s, rank))) + 0.1, dtype=tensor.dtype) for s in tensor.shape]
    grams = [f.T @ f for f in factors]
    tensor_norm_sq = float(xp.sum(tensor * tensor))

    penalised = sparsity > 0.0 or orthogonality > 0.0
    off_diag = xp.asarray(np.ones((rank, rank)) - np.eye(rank), dtype=tensor.dtype) if orthogonality > 0.0 else None

    prev_error = math.inf
    for _ in range(n_iter_max):
        for mode in range(3):
            other = [i for i in range(3) if i != mode]
            numerator = xp.einsum(
                _MTTKRP_EINSUM[mode],
                tensor,
                factors[other[0]],
                factors[other[1]],
                optimize=True,
            )
            gram_prod = grams[other[0]] * grams[other[1]]
            denominator = factors[mode] @ gram_prod
            if mode == 1:  # variable factor B — apply interpretability penalties
                if sparsity > 0.0:
                    # Relative L1: a uniform offset sized to the mean explained magnitude so the
                    # penalty stays meaningful regardless of factor scale / tensor dimensions.
                    denominator = denominator + sparsity * float(xp.mean(denominator))
                if orthogonality > 0.0:
                    denominator = denominator + orthogonality * (factors[mode] @ off_diag)
            factors[mode] = factors[mode] * numerator / (denominator + eps)
            grams[mode] = factors[mode].T @ factors[mode]

        inner = float(xp.sum(factors[2] * numerator))
        gram_all = grams[0] * grams[1] * grams[2]
        err_sq = max(tensor_norm_sq - 2 * inner + float(xp.sum(gram_all)), 0.0)
        error = math.sqrt(err_sq) / max(math.sqrt(tensor_norm_sq), eps)

        if abs(prev_error - error) < tol:
            break
        prev_error = error

        # Fix the scaling gauge: normalise the patient (A) and temporal (C) factors to unit columns,
        # pushing their scale into B. This makes the other modes' Gram matrices O(1) so the penalty
        # strengths on B are on a consistent, interpretable scale instead of being defeated by — or
        # swamped by — CP's freedom to trade magnitude between the three factors.
        if penalised:
            for m in (0, 2):
                norms = xp.clip(xp.linalg.norm(factors[m], axis=0), min=eps)
                factors[m] = factors[m] / norms[None, :]
                factors[1] = factors[1] * norms[None, :]
            grams = [f.T @ f for f in factors]

    weights = xp.ones(rank, dtype=tensor.dtype)
    for i in range(3):
        norms = xp.clip(xp.linalg.norm(factors[i], axis=0), min=eps)
        weights = weights * norms
        factors[i] = factors[i] / norms[None, :]

    return weights, factors


def ncp(
    edata: EHRData,
    *,
    layer: str,
    rank: int = 4,
    n_iter_max: int = 300,
    sigmoid_transform: bool = False,
    sparsity: float = 0.0,
    orthogonality: float = 0.0,
    key_added: str = "ncp",
    random_state: int = 0,
    copy: bool = False,
) -> EHRData | None:
    r"""Non-negative CP (PARAFAC) decomposition of a 3D temporal EHR layer.

    CP (CANDECOMP/PARAFAC) decomposition factorises a 3-way tensor
    :math:`X \in \mathbb{R}^{I \times J \times K}` into a sum of ``rank``
    outer products:

    .. math::

        X \approx \sum_r a_r \otimes b_r \otimes c_r

    where each triplet :math:`(a_r, b_r, c_r)` is a *component*:

    * :math:`a_r \in \mathbb{R}^I` — **patient factor**: how strongly each observation expresses component *r*.
    * :math:`b_r \in \mathbb{R}^J` — **variable factor**: which clinical variables are characteristic of component *r*.
    * :math:`c_r \in \mathbb{R}^K` — **temporal factor**: how the pattern of component *r* evolves over the time axis.

    The *Non-negative* variant (NCP) constrains all factors to be :math:`\geq 0`,
    which is natural for count-like or probability data and yields parts-based, interpretable components (analogous to NMF for matrices).

    Factors are estimated by Multiplicative Updates (Lee & Seung, 2001).
    Each factor matrix is updated in closed form while the others are held fixed, cycling through the three modes until convergence:

    .. math::

        F_{\text{mode}} \leftarrow F_{\text{mode}} \odot
        \frac{\mathcal{X}_{(\text{mode})} \, \mathrm{KR}(F_{-\text{mode}})}
             {F_{\text{mode}} \, \mathrm{KR}(F_{-\text{mode}})^\top
              \mathrm{KR}(F_{-\text{mode}}) + \varepsilon}

    where :math:`\mathcal{X}_{(\text{mode})}` is the mode-*n* matricisation of the tensor and :math:`\mathrm{KR}` denotes the Khatri–Rao product of the remaining factor matrices.

    Args:
        edata: Central data object.
        layer: Key of the 3D layer to decompose (shape ``n_obs × n_vars × n_time``).
            All values must be non-negative (use ``sigmoid_transform=True`` for logit layers, or ``np.abs`` / clipping beforehand).
        rank: Number of components (rank of the decomposition).
            Each component describes one co-occurring patient sub-group, variable signature, and temporal trajectory.
        n_iter_max: Maximum number of multiplicative-update iterations.
            300 is sufficient for most datasets; increase if the error has not converged (check ``edata.uns[key_added]["params"]``).
        sigmoid_transform: If ``True``, apply a sigmoid transformation to the layer before decomposition.
            Useful when the layer contains raw logits.
        sparsity: L1 penalty strength on the **variable factor** :math:`B` (default ``0`` = off).
            Larger values push small variable loadings to exactly zero, so each clinical variable loads
            onto *few* components instead of being spread thinly across many. This sharpens the
            "this program = these diseases" interpretation. Start small (e.g. ``0.01``–``0.1`` relative to
            the layer scale) and increase until the loadings are as discrete as the interpretation needs.
        orthogonality: Penalty strength on the off-diagonal entries of :math:`B^\top B` (default ``0`` = off).
            Larger values discourage two components from sharing the same variables, yielding programs with
            *fewer, more distinct features each*. Use when you want near-disjoint variable signatures across
            components; it can fight the data when variables genuinely co-occur across programs, so raise it
            gradually. Can be combined with ``sparsity``.
        key_added: Key prefix for storing results. Results are stored as:

            * ``edata.obsm["X_{key_added}"]`` — patient factors, shape ``(n_obs, rank)``.
            * ``edata.varm["{key_added}_loadings"]`` — variable factors, shape ``(n_vars, rank)``.
            * ``edata.uns["{key_added}"]["temporal_factors"]`` — temporal factors, shape ``(n_time, rank)``.
        random_state: Random seed for the factor initialisation.
        copy: Whether to return a copy rather than modifying in place.

    Returns:
        ``None`` if ``copy=False``, else a modified copy of ``edata``.

    Examples:
        >>> import ehrdata as ed, ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=8, n_centers=3, n_observations=30, base_timepoints=12)
        >>> ep.tl.ncp(edata, layer="tem_data", rank=3, sigmoid_transform=True)
        >>> edata.obsm["X_ncp"].shape
        (30, 3)
        >>> edata.varm["ncp_loadings"].shape
        (8, 3)
        >>> edata.uns["ncp"]["temporal_factors"].shape
        (12, 3)
    """
    if layer not in edata.layers:
        raise KeyError(f"Layer {layer!r} not found in edata.layers. Available: {list(edata.layers)}")

    tensor = np.asarray(edata.layers[layer], dtype=np.float64)
    if tensor.ndim != 3:
        raise ValueError(f"Layer {layer!r} must be 3D (n_obs × n_vars × n_time), got shape {tensor.shape}.")

    if sigmoid_transform:
        from scipy.special import expit

        tensor = expit(tensor)

    edata = edata.copy() if copy else edata

    weights, factors = _nonneg_cp(
        tensor,
        rank=rank,
        n_iter_max=n_iter_max,
        random_state=random_state,
        sparsity=sparsity,
        orthogonality=orthogonality,
    )
    A, B, C = factors
    A = A * weights[None, :]

    edata.obsm[f"X_{key_added}"] = A  # (n_obs, rank)
    edata.varm[f"{key_added}_loadings"] = B  # (n_vars, rank)
    edata.uns[key_added] = {
        "params": {
            "layer": layer,
            "rank": rank,
            "n_iter_max": n_iter_max,
            "sigmoid_transform": sigmoid_transform,
            "sparsity": sparsity,
            "orthogonality": orthogonality,
        },
        "temporal_factors": C,  # (n_time, rank)
    }

    return edata if copy else None
