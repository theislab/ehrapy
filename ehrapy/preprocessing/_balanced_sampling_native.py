from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from anndata import AnnData

from anndata import AnnData

import numpy as np

def random_resample(
        label: str,
        target: str = "balanced",
        method: Literal["under", "over"] = "under",
        random_state: int = 0
        ) -> tuple[np.ndarray, np.ndarray]:
    """Under- or over-sample the data to achieve a balanced dataset.
    Args:
        label: The labels of the data.
        target: The target number of samples for each class. If "balanced", it will balance the classes to the minimum class size.
        method: The sampling method, either "under" for under-sampling or "over" for over-sampling.
        random_state: Random seed.
    Returns:
        A tuple of (sampled_indices, sampled_labels).
    """
    label = np.asarray(label)
    rnd = np.random.default_rng(random_state)
    classes, counts = np.unique(label, return_counts=True)

    if target == "balanced":
        if method== "under":
            target = counts.min()
        elif method == "over":
            target = counts.max()
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
    indices = []

    for c in classes:
        class_idx = np.where(label == c)[0]
        n = len(class_idx)
        if method == "under":
            if n > target:
                sampled_idx = rnd.choice(class_idx, size=target, replace=False)
                indices.extend(sampled_idx)
            else:
                indices.extend(class_idx)
        elif method == "over":
            if n < target:
                sampled_idx = rnd.choice(class_idx, size=target, replace=True)
                indices.extend(sampled_idx)
            else:
                indices.extend(class_idx)

    sample_indices = np.array(indices)
    return sample_indices, label[sample_indices]

def resample_adata(
        adata: AnnData,
        key: str,
        method: Literal["under", "over", "smote"] = "under",
        random_state: int = 0,
        copy: bool = False
        ) -> AnnData:
    """Resample an AnnData object based on a key in obs.
    Args:
        adata: The annotated data matrix of shape `n_obs` × `n_vars`.
        key: The key in `adata.obs` that contains the group information.
        method: The sampling method, either "under" for under-sampling or "over" for over-sampling.
        random_state: Random seed.
        copy: If True, return a copy of the balanced data.
    Returns:
        A new `AnnData` object with the resampled balanced data.
    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.data.diabetes_130_fairlearn(columns_obs_only=["age"])
        >>> adata.obs.age.value_counts()
        age
        'Over 60 years'          68541
        '30-60 years'            30716
        '30 years or younger'     2509
        >>> adata_balanced = ep.pp.sample(adata, key="age", method="under", random_state=42)
        >>> adata_balanced.obs.age.value_counts()
        age
        '30 years or younger'    2509
        '30-60 years'            2509
        'Over 60 years'          2509
    """

    if not isinstance(adata, AnnData):
        raise ValueError(f"Input data is not an AnnData object: type of {adata} is {type(adata)}")
    
    if key not in adata.obs:
        raise ValueError(f"Key '{key}' not found in adata.obs")
    
    labels = adata.obs[key].values

    if method == "under" or method == "over":
        sampled_indices, sampled_labels = random_resample(labels, method=method, random_state=random_state)
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    if copy:
        return adata[sampled_indices].copy()
    else:
        adata._inplace_subset_obs(sampled_indices)
