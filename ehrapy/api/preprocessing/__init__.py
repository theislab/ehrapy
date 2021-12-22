from typing import Dict, Optional, Tuple, Union

from anndata import AnnData

from ehrapy.api.preprocessing._data_imputation import Imputation
from ehrapy.api.preprocessing.encoding import encode, type_overview, undo_encoding


def impute(
    adata: AnnData,
    mode: str,
    copy: bool = False,
    **kwargs
) -> Optional[AnnData]:
    """Replaces all missing values in all or the specified columns with the passed value

    Args:
        adata: :class:`~anndata.AnnData` object containing X to impute values in
        mode: TODO
        copy: Whether to return a copy or act in place

    Returns:
        :class:`~anndata.AnnData` object with imputed X

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            TODO
    """
    impute_modes = {"explicit": Imputation.explicit,
                    "knn": Imputation.knn,
                    "mean": Imputation.mean}
    return impute_modes.get(mode)(adata, copy, **kwargs)


from ehrapy.api.preprocessing._scanpy_pp_api import *  # noqa: E402,F403
