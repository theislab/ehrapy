from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import ehrapy as ep

if TYPE_CHECKING:
    from anndata import AnnData
    from ehrdata import EHRData

from ehrapy._compat import use_ehrdata


@use_ehrdata(deprecated_after="1.0.0")
def balanced_sample(
    edata: EHRData | AnnData,
    *,
    key: str,
    random_state: int = 0,
    method: Literal["RandomUnderSampler", "RandomOverSampler"] = "RandomUnderSampler",
    sampler_kwargs: dict = None,
    copy: bool = False,
) -> EHRData | AnnData:
    warnings.warn(
        "This function is renamed. Use ep.pp.sample instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ep.pp.sample(
        data=edata, rng=random_state, balanced=True, balanced_method=method, balanced_key=key, copy=copy
    )
