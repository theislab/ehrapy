from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ehrapy._compat import use_ehrdata

if TYPE_CHECKING:
    from anndata import AnnData
    from ehrdata import EHRData


def _khatri_rao(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Khatri-Rao (column-wise Kronecker) product of two matrices."""
    I, R = A.shape
    J, _ = B.shape
    return (A[:, np.newaxis, :] * B[np.newaxis, :, :]).reshape(I * J, R)


def _unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Mode-n unfolding (matricisation) of a tensor."""
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def _nonneg_cp(
    tensor: np.ndarray,
    rank: int,
    n_iter_max: int = 300,
    tol: float = 1e-8,
    random_state: int = 0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Non-negative CP decomposition via Multiplicative Updates (Lee & Seung).

    Parameters
    ----------
    tensor
        Non-negative input tensor of arbitrary order (≥2).
    rank
        Number of components.
    n_iter_max
        Maximum number of iterations.
    tol
        Convergence tolerance on the relative change in reconstruction error.
    random_state
        Seed for initialisation.

    Returns:
    -------
    weights
        Per-component weights of shape ``(rank,)``.
    factors
        List of non-negative factor matrices, one per tensor mode.
    """
    rng = np.random.default_rng(random_state)
    ndims = tensor.ndim
    eps = 1e-12

    factors = [np.abs(rng.standard_normal((s, rank))) + 0.1 for s in tensor.shape]

    prev_error = np.inf
    for _ in range(n_iter_max):
        for mode in range(ndims):
            others = [f for i, f in enumerate(factors) if i != mode]
            kr = others[-1]
            for f in reversed(others[:-1]):
                kr = _khatri_rao(f, kr)

            X_mode = _unfold(tensor, mode)
            numerator = X_mode @ kr
            denominator = factors[mode] @ (kr.T @ kr) + eps
            factors[mode] *= numerator / denominator

        recon = np.zeros_like(tensor)
        for r in range(rank):
            component = factors[0][:, r]
            for f in factors[1:]:
                component = np.multiply.outer(component, f[:, r])
            recon += component
        error = np.linalg.norm(tensor - recon) / max(np.linalg.norm(tensor), eps)

        if abs(prev_error - error) < tol:
            break
        prev_error = error

    weights = np.ones(rank)
    for i in range(ndims):
        norms = np.maximum(np.linalg.norm(factors[i], axis=0), eps)
        weights *= norms
        factors[i] /= norms[np.newaxis, :]

    return weights, factors


def ncp(
    edata: EHRData,
    *,
    layer: str,
    rank: int = 4,
    n_iter_max: int = 300,
    init: Literal["random"] = "random",
    sigmoid_transform: bool = False,
    key_added: str = "ncp",
    random_state: int = 0,
    copy: bool = False,
) -> EHRData | None:
    r"""Non-negative CP (PARAFAC) decomposition of a 3D temporal layer.

    Decomposes the stored 3D data into three non-negative factor matrices
    using multiplicative updates.

    Args:
        edata: Central data object.
        layer: Key of the 3D layer to decompose (shape ``n_obs × n_vars × n_time``).
        rank: Number of components (rank of the decomposition).
        n_iter_max: Maximum number of multiplicative-update iterations.
        init: Initialisation strategy (currently only ``"random"``).
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
        >>> import numpy as np
        >>> import ehrdata as ed, ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=8, n_centers=3, n_observations=30, base_timepoints=12)
        >>> edata.layers["tem_data"] = np.abs(edata.layers["tem_data"])
        >>> ep.tl.ncp(edata, layer="tem_data", rank=3)
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

    weights, factors = _nonneg_cp(tensor, rank=rank, n_iter_max=n_iter_max, random_state=random_state)
    A, B, C = factors
    A = A * weights[np.newaxis, :]

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
    }

    return edata if copy else None
