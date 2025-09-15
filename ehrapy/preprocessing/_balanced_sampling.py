from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import ehrapy as ep

if TYPE_CHECKING:
    from anndata import AnnData


def balanced_sample(
    adata: AnnData,
    *,
    key: str,
    random_state: int = 0,
    method: Literal["under", "over"] = "under",
    sampler_kwargs: dict = None,
    copy: bool = False,
) -> AnnData:
    warnings.warn(
        "This function is renamed. Use ep.pp.sample instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ep.pp.sample(
        data=adata, rng=random_state, balanced=True, balanced_method=method, balanced_key=key, copy=copy
    )
