from typing import List, Union

from anndata import AnnData
from mudata import MuData

from ehrapy.api import ehrapy_settings
from ehrapy.api.io._read import read
from ehrapy.api.preprocessing.encoding import encode


def mimic_2(encoded: bool = False) -> AnnData:  # pragma: no cover
    """Loads the MIMIC-II dataset

    Args:
        encoded: Whether to return an already encoded object

    Returns:
        :class:`~anndata.AnnData` object of the MIMIC-II dataset

    Example:
        .. code-block:: python

        import ehrapy.api as ep
        adata = eh.data.mimic_2(encode=True)
    """
    adata = read(
        dataset_path=f"{ehrapy_settings.datasetdir}/ehrapy_mimic2.csv",
        download_dataset_name="ehrapy_mimic2.csv",
        backup_url="https://www.physionet.org/files/mimic2-iaccd/1.0/full_cohort_data.csv?download",
        suppress_warnings=True,
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def mimic_3_demo(encoded: bool = False, mudata: bool = False) -> Union[MuData, List[AnnData]]:  # pragma: no cover
    """Loads the MIMIC-III demo dataset

    Args:
        encoded: Whether to return an already encoded object
        mudata: Whether to return a MuData object. Returns a Dictionary of file names to AnnData objects if False

    Returns:
        :class:`~mudata.MuData` object of the MIMIC-III demo Dataset

    Example:
    .. code-block:: python

        import ehrapy.api as ep
        mudata = eh.data.mimic_3_demo(encode=True, return_mudata=True)
    """
    mdata = read(
        dataset_path=f"{ehrapy_settings.datasetdir}/ehrapy_mimic_3",
        download_dataset_name="ehrapy_mimic_3",
        backup_url="https://physionet.org/static/published-projects/mimiciii-demo/mimic-iii-clinical-database-demo-1.4.zip",
        return_mudata=mudata,
        extension="csv",
    )
    if encoded:
        if not mudata:
            raise ValueError(
                "Currently we only support the encoding of a single AnnData object or a single MuData object."
            )
        encode(mdata, autodetect=True)

    return mdata
