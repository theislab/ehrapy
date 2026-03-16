from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ehrapy._compat import use_ehrdata

if TYPE_CHECKING:
    from anndata import AnnData
    from ehrdata import EHRData


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

    Uses ``tensorly``'s ``non_negative_parafac`` (required dependency).

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
        >>> import numpy as np, pandas as pd
        >>> import ehrdata as ed, ehrapy as ep
        >>> np.random.seed(0)
        >>> tensor = np.abs(np.random.randn(30, 8, 12))  # patients × vars × time
        >>> edata = ed.EHRData(
        ...     shape=(30, 8),
        ...     layers={"data": tensor},
        ...     var=pd.DataFrame(index=[f"var_{i}" for i in range(8)]),
        ... )
        >>> ep.tl.ncp(edata, layer="data", rank=3)
        >>> edata.obsm["X_ncp"].shape  # (30, 3)  – sample factors
        >>> edata.varm["ncp_loadings"].shape  # (8, 3)   – variable factors
        >>> edata.uns["ncp"]["temporal_factors"].shape  # (12, 3) – time factors
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

    from tensorly.decomposition import non_negative_parafac

    weights, factors = non_negative_parafac(
        tensor, rank=rank, init=init, n_iter_max=n_iter_max, random_state=random_state
    )
    A, B, C = (np.asarray(f) for f in factors)
    # absorb weights into the sample factor so each component is self-contained
    A = A * np.asarray(weights)[np.newaxis, :]

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
