from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

from anndata import AnnData
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def sample(
    adata: AnnData,
    key: str,
    random_state: int = 0,
    method: str = "RandomUnderSampler",
    sampler_kwargs: dict = None,
) -> AnnData:
    """Balancing groups in the dataset.

    Balancing groups in the dataset based on group members in `.obs[key]` using the [imbalanced-learn](https://imbalanced-learn.org/stable/index.html) package.
    Currently supports `RandomUnderSampler` and `RandomOverSampler`.

    Args:
        adata: The annotated data matrix of shape `n_obs` × `n_vars`.
        key: The key in `adata.obs` that contains the group information.
        random_state: Random seed. Defaults to 0.
        method: The method to use for balancing. Defaults to "RandomUnderSampler".
        sampler_kwargs: Keyword arguments for the sample, see the `imbalanced-learn` documentation for options. Defaults to None.

    Returns:
        A new `AnnData` object, with the balanced groups.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.data.diabetes_130_fairlearn(columns_obs_only=["age"])
        >>> print("distribution of age groups:\n", adata.obs.age.value_counts())
        >>> adata_balanced = ep.pp.sample(adata, key="age")
        >>> print(
        ...     "distribution of age groups after undersampling:\n",
        ...     adata_balanced.obs.age.value_counts(),
        ... )

        >>> distribution of groups:
        >>>  age
        >>> 'Over 60 years'          68541
        >>> '30-60 years'            30716
        >>> '30 years or younger'     2509
        >>> distribution of groups after undersampling:
        >>>  age
        >>> '30 years or younger'    2509
        >>> '30-60 years'            2509
        >>> 'Over 60 years'          2509

    """

    if not isinstance(adata, AnnData):
        raise ValueError(f"Input data is not an AnnData object: type of {adata}, is {type(adata)}")

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

    if key in adata.obs.keys():
        use_label = adata.obs[key]
    else:
        raise ValueError(f"key not in adata.obs: {key}")

    sampler.fit_resample(adata.X, use_label)

    adata_sampled = adata[sampler.sample_indices_, :].copy()

    # results computed from data should be recomputed if the data changes
    del adata_sampled.obsm
    del adata_sampled.varm
    del adata_sampled.uns
    del adata_sampled.obsp
    del adata_sampled.varp

    return adata_sampled
