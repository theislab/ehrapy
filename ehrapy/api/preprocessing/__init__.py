from ehrapy.api.preprocessing._scanpy_pp_api import *  # noqa: E402,F403
from ehrapy.api.preprocessing.encoding import encode, type_overview, undo_encoding

from typing import Optional

from anndata import AnnData

from ehrapy.api.preprocessing._data_imputation import knn, mean, _replace_explicit
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
        mode: Imputation procedure to use
        copy: Whether to return a copy or act in place

    Returns:
        :class:`~anndata.AnnData` object with imputed X

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            TODO
    """
    if copy:
        adata = adata.copy()

    impute_modes = {"explicit": _replace_explicit,
                    "knn": knn,
                    "mean": mean}
    return impute_modes.get(mode)(adata, copy, **kwargs)
