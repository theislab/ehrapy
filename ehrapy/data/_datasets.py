from __future__ import annotations

from anndata import AnnData
from mudata import MuData

from ehrapy import ehrapy_settings
from ehrapy.io._read import read
from ehrapy.preprocessing.encoding._encode import encode


def mimic_2(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:
    """Loads the MIMIC-II dataset.

    More details: https://physionet.org/content/mimic2-iaccd/1.0/

    Args:
        encoded: Whether to return an already encoded object
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the MIMIC-II dataset

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.dt.mimic_2(encode=True)
    """
    adata = read(
        dataset_path=f"{ehrapy_settings.datasetdir}/ehrapy_mimic2.csv",
        download_dataset_name="ehrapy_mimic2.csv",
        backup_url="https://www.physionet.org/files/mimic2-iaccd/1.0/full_cohort_data.csv?download",
        columns_obs_only=columns_obs_only,
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def mimic_3_demo(
    encoded: bool = False,
    mudata: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> MuData | list[AnnData]:
    """Loads the MIMIC-III demo dataset.

    Args:
        encoded: Whether to return an already encoded object
        mudata: Whether to return a MuData object. Returns a Dictionary of file names to AnnData objects if False

    Returns:
        :class:`~mudata.MuData` object of the MIMIC-III demo Dataset

    Example:
        .. code-block:: python

            import ehrapy as ep

            adatas = ep.dt.mimic_3_demo(encode=True)
    """
    mdata = read(
        dataset_path=f"{ehrapy_settings.datasetdir}/ehrapy_mimic_3",
        download_dataset_name="ehrapy_mimic_3",
        backup_url="https://physionet.org/static/published-projects/mimiciii-demo/mimic-iii-clinical-database-demo-1.4.zip",
        columns_obs_only=columns_obs_only,
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


def heart_failure(columns_obs_only: dict[str, list[str]] | list[str] | None = None) -> AnnData:
    """Loads the heart failure dataset.

    More details: http://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
    Preprocessing: https://github.com/theislab/ehrapy-datasets/tree/main/heart_failure
    This dataset only contains numericals and therefore does not need any encoding.

    Args:
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the MIMIC-II dataset

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.dt.heart_failure()
    """
    adata = read(
        dataset_path=f"{ehrapy_settings.datasetdir}/heart_failure.csv",
        download_dataset_name="heart_failure.csv",
        backup_url="https://figshare.com/ndownloader/files/33952934",
        columns_obs_only=columns_obs_only,
        extension="csv",
        index_column="patient_id",
    )

    return adata


def diabetes_130(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:
    """Loads the diabetes-130 dataset

    More details: http://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
    Preprocessing: https://github.com/theislab/ehrapy-datasets/tree/main/diabetes_130/diabetes_130.ipynb

    Args:
        encoded: Whether to return an already encoded object
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Diabetes 130 dataset

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.dt.diabetes_130(encode=True)
    """
    adata = read(
        dataset_path=f"{ehrapy_settings.datasetdir}/diabetes_130.csv",
        download_dataset_name="diabetes_130.csv",
        backup_url="https://figshare.com/ndownloader/files/33950546",
        columns_obs_only=columns_obs_only,
        extension="csv",
        index_column="encounter_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def chronic_kidney_disease(
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:  # pragma: no cover
    """Loads the Chronic Kidney Disease dataset

    More details: https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease
    Preprocessing: https://github.com/theislab/ehrapy-datasets/tree/main/chronic_kidney_disease/chronic_kidney_disease.ipynb

    Args:
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Chronic Kidney Disease dataset

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.data.chronic_kidney_disease()
    """
    adata = read(
        dataset_path=f"{ehrapy_settings.datasetdir}/chronic_kidney_disease_precessed.csv",
        download_dataset_name="chronic_kidney_disease.csv",
        backup_url="https://figshare.com/ndownloader/files/33989261",
        columns_obs_only=columns_obs_only,
        extension="csv",
        index_column="Patient_id",
    )

    return adata
