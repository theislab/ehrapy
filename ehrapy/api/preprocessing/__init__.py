from anndata import AnnData

from ehrapy.api.preprocessing._data_imputation import (
    _knn,
    _mean,
    _median,
    _miss_forest,
    _most_frequent,
    _replace_explicit,
)
from ehrapy.api.preprocessing._quality_control import calculate_qc_metrics
from ehrapy.api.preprocessing._scanpy_pp_api import *  # noqa: E402,F403


def impute(adata: AnnData, mode: str, copy: bool = False, **kwargs) -> AnnData:
    """Replaces all missing values in all or the specified columns with the passed value

    Args:
        adata: :class:`~anndata.AnnData` object containing X to impute values in
        mode: Imputation procedure to use
        copy: Whether to return a copy or act in place

    Returns:
        :class:`~anndata.AnnData` object with imputed X

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2()
            TODO
    """
    if copy:
        adata = adata.copy()

    impute_modes = {
        "explicit": _replace_explicit,
        "knn": _knn,
        "mean": _mean,
        "median": _median,
        "most_frequent": _most_frequent,
        "miss_forest": _miss_forest,
    }
    return impute_modes.get(mode)(adata, **kwargs)
