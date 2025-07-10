from __future__ import annotations

from typing import Literal

from anndata import AnnData
from ehrdata import EHRData
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

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
) -> EHRData | AnnData | None:
    """Balancing groups in the dataset.

    Balancing groups in the dataset based on group members in `.obs[key]` using the `imbalanced-learn <https://imbalanced-learn.org/stable/index.html>`_ package.
    Currently, supports `RandomUnderSampler` and `RandomOverSampler`.

    Note that `RandomOverSampler <https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html>`_
    only replicates observations of the minority groups, which distorts several downstream analyses, very prominently neighborhood calculations and downstream analyses depending on that.
    The `RandomUnderSampler <https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html>`_
    by  default undersamples the majority group without replacement, not causing this issues of replicated observations.

    Args:
        edata: The annotated data matrix of shape `n_obs` × `n_vars`.
        key: The key in `edata.obs` that contains the group information.
        random_state: Random seed.
        method: The method to use for balancing.
        sampler_kwargs: Keyword arguments for the sampler, see the `imbalanced-learn` documentation for options.
        copy: If True, return a copy of the balanced data.

    Returns:
        A new `AnnData` object, with the balanced groups.

    Examples:
        >>> import ehrapy as ep
        >>> edata = ep.data.diabetes_130_fairlearn(columns_obs_only=["age"])
        >>> edata.obs.age.value_counts()
        age
        'Over 60 years'          68541
        '30-60 years'            30716
        '30 years or younger'     2509
        >>> edata_balanced = ep.pp.sample(edata, key="age")
        >>> edata_balanced.obs.age.value_counts()
        age
        '30 years or younger'    2509
        '30-60 years'            2509
        'Over 60 years'          2509
    """
    if not isinstance(edata, EHRData | AnnData):
        raise ValueError(f"Input data is not an EHRData orAnnData object: type of {edata}, is {type(edata)}")

    if sampler_kwargs is None:
        sampler_kwargs = {"random_state": random_state}
    else:
        sampler_kwargs["random_state"] = random_state

    if method == "RandomUnderSampler":
        sampler = RandomUnderSampler(**sampler_kwargs)
    elif method == "RandomOverSampler":
        sampler = RandomOverSampler(**sampler_kwargs)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    if key in edata.obs.keys():
        use_label = edata.obs[key]
    else:
        raise ValueError(f"key not in edata.obs: {key}")

    sampler.fit_resample(edata.X, use_label)

    if copy:
        return edata[sampler.sample_indices_].copy()
    else:
        edata._inplace_subset_obs(sampler.sample_indices_)
        return None
