from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import ehrapy as ep

if TYPE_CHECKING:
    from ehrdata import EHRData


def balanced_sample(
    edata: EHRData,
    *,
    key: str,
    random_state: int = 0,
    method: Literal["RandomUnderSampler", "RandomOverSampler"] = "RandomUnderSampler",
    sampler_kwargs: dict = None,
    copy: bool = False,
) -> EHRData:
    warnings.warn(
        "This function is renamed. Use ep.pp.sample instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ep.pp.sample(
        data=edata, rng=random_state, balanced=True, balanced_method=method, balanced_key=key, copy=copy
    )
