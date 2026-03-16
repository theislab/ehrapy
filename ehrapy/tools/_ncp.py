from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ehrapy._compat import use_ehrdata

if TYPE_CHECKING:
    from anndata import AnnData
    from ehrdata import EHRData


# ── Internal numpy NCP-ALS (no external dep) ──────────────────────────────────


def _khatri_rao(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Column-wise Kronecker (Khatri-Rao) product: (I, R) × (J, R) → (I·J, R)."""
    return np.array([np.kron(A[:, r], B[:, r]) for r in range(A.shape[1])]).T


def _ncp_als(
    tensor: np.ndarray,
    rank: int,
    n_iter_max: int = 300,
    tol: float = 1e-7,
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Non-negative CP decomposition via Alternating Least Squares (numpy-only)."""
    I, J, K = tensor.shape
    rng = np.random.default_rng(random_state)
    A = rng.random((I, rank)) + 0.1
    B = rng.random((J, rank)) + 0.1
    C = rng.random((K, rank)) + 0.1

    X1 = tensor.reshape(I, J * K)
    X2 = tensor.transpose(1, 0, 2).reshape(J, I * K)
    X3 = tensor.transpose(2, 0, 1).reshape(K, I * J)

    errors: list[float] = []
    for it in range(n_iter_max):
        A = np.maximum(X1 @ _khatri_rao(C, B) @ np.linalg.pinv((C.T @ C) * (B.T @ B)), 1e-12)
        B = np.maximum(X2 @ _khatri_rao(C, A) @ np.linalg.pinv((C.T @ C) * (A.T @ A)), 1e-12)
        C = np.maximum(X3 @ _khatri_rao(B, A) @ np.linalg.pinv((B.T @ B) * (A.T @ A)), 1e-12)
        recon = sum(np.outer(A[:, r], np.outer(B[:, r], C[:, r]).ravel()).reshape(I, J, K) for r in range(rank))
        err = float(np.linalg.norm(tensor - recon) / (np.linalg.norm(tensor) + 1e-12))
        errors.append(err)
        if it > 10 and abs(errors[-2] - errors[-1]) < tol:
            break

    return A, B, C, errors


# ── Public API ─────────────────────────────────────────────────────────────────


@use_ehrdata(deprecated_after="1.0.0")
def ncp(
    edata: EHRData | AnnData,
    *,
    layer: str,
    rank: int = 4,
    n_iter_max: int = 300,
    init: str = "random",
    sigmoid_transform: bool = False,
    key_added: str = "ncp",
    random_state: int = 0,
    copy: bool = False,
) -> EHRData | AnnData | None:
    r"""Non-negative CP (PARAFAC) decomposition of a 3D temporal layer.

    Decomposes the tensor :math:`\\mathcal{X} \\approx \\sum_{r=1}^{R} \\mathbf{a}_r \\otimes \\mathbf{b}_r \\otimes \\mathbf{c}_r`
    (all factors non-negative) and stores the three factor matrices in
    ``.obsm``, ``.varm``, and ``.uns``.

    Uses ``tensorly``'s ``non_negative_parafac`` when available; otherwise falls
    back to a built-in numpy ALS implementation.

    Args:
        edata: Central data object.
        layer: Key of the 3D layer to decompose (shape ``n_obs × n_vars × n_time``).
        rank: Number of components (rank of the decomposition).
        n_iter_max: Maximum number of ALS iterations.
        init: Initialisation strategy passed to tensorly (``"random"`` or ``"svd"``).
        sigmoid_transform: If ``True``, apply a sigmoid transformation to the layer
            before decomposition. Useful when the layer contains raw logits.
        key_added: Key prefix for storing results. Results are stored as
            ``edata.obsm["X_{key_added}"]`` (sample factors, shape ``n_obs × rank``),
            ``edata.varm["{key_added}_loadings"]`` (variable factors, shape ``n_vars × rank``),
            and ``edata.uns["{key_added}"]`` (temporal factors + metadata).
        random_state: Random seed for reproducibility.
        copy: Whether to return a copy rather than modifying in place.

    Returns:
        ``None`` if ``copy=False``, else a modified copy of ``edata``.

    Examples:
        >>> import ehrapy as ep
        >>> import ehrdata as ed
        >>> edata = ed.dt.physionet2019(layer="tem_data", n_samples=200)
        >>> ep.tl.ncp(edata, layer="tem_data", rank=3)
        >>> edata.obsm["X_ncp"]  # sample factors  (n_obs × rank)
        >>> edata.varm["ncp_loadings"]  # variable factors (n_vars × rank)
        >>> edata.uns["ncp"]["temporal_factors"]  # time factors (n_time × rank)
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

    try:
        from tensorly.decomposition import non_negative_parafac

        weights, factors = non_negative_parafac(
            tensor, rank=rank, init=init, n_iter_max=n_iter_max, random_state=random_state
        )
        A, B, C = (np.asarray(f) for f in factors)
        # absorb weights into the sample factor so each component is self-contained
        A = A * np.asarray(weights)[np.newaxis, :]
        errors: list[float] = []
    except ImportError:
        A, B, C, errors = _ncp_als(tensor, rank=rank, n_iter_max=n_iter_max, random_state=random_state)

    edata.obsm[f"X_{key_added}"] = A  # (n_obs, rank)
    edata.varm[f"{key_added}_loadings"] = B  # (n_vars, rank)
    edata.uns[key_added] = {
        "params": {
            "layer": layer,
            "rank": rank,
            "n_iter_max": n_iter_max,
            "init": init,
            "sigmoid_transform": sigmoid_transform,
        },
        "temporal_factors": C,  # (n_time, rank)
        "reconstruction_errors": errors,
    }

    return edata if copy else None
