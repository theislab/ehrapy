from __future__ import annotations

import pandas as pd
from anndata import AnnData

from ehrapy import ehrapy_settings
from ehrapy.io._read import read_csv, read_fhir, read_h5ad
from ehrapy.preprocessing._encode import encode


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

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
    """
    adata = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/ehrapy_mimic2.csv",
        download_dataset_name="ehrapy_mimic2.csv",
        backup_url="https://www.physionet.org/files/mimic2-iaccd/1.0/full_cohort_data.csv?download",
        columns_obs_only=columns_obs_only,
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def mimic_2_preprocessed() -> AnnData:
    """Loads the preprocessed MIMIC-II dataset.

    More details: https://physionet.org/content/mimic2-iaccd/1.0/
    The dataset was preprocessed according to: https://github.com/theislab/ehrapy-datasets/tree/main/mimic_2

    Returns:
        :class:`~anndata.AnnData` object of the prprocessed MIMIC-II dataset

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2_preprocessed()
    """
    adata = read_h5ad(
        dataset_path=f"{ehrapy_settings.datasetdir}/ehrapy_mimic2.csv",
        download_dataset_name="ehrapy_mimic_2_preprocessed.h5ad",
        backup_url="https://figshare.com/ndownloader/files/39727936",
    )

    return adata


def mimic_3_demo(
    encoded: bool = False,
    anndata: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> dict[str, AnnData] | dict[str, pd.DataFrame]:
    """Loads the MIMIC-III demo dataset as a dictionary of Pandas DataFrames.

    The MIMIC-III dataset comes in the form of 26 CSV tables. Although, it is possible to return one AnnData object per
    csv table, it might be easier to start with Pandas DataFrames to aggregate the desired measurements with Pandas SQL.
    https://github.com/yhat/pandasql/ might be useful.
    The resulting DataFrame can then be transformed into an AnnData object with :func:`~ehrapy.anndata.df_to_anndata`.

    Args:
        encoded: Whether to return an already encoded object.
        anndata: Whether to return one AnnData object per CSV file. Defaults to False
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        A dictionary of AnnData objects or a dictionary of Pandas DataFrames

    Examples:
        >>> import ehrapy as ep
        >>> dfs = ep.dt.mimic_3_demo()
    """
    data = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/ehrapy_mimic_3",
        download_dataset_name="ehrapy_mimic_3",
        backup_url="https://physionet.org/static/published-projects/mimiciii-demo/mimic-iii-clinical-database-demo-1.4.zip",
        return_dfs=False if anndata else True,
        columns_obs_only=columns_obs_only,
        archive_format="zip",
    )
    if encoded:
        if not anndata:
            raise ValueError("Can only encode AnnData objects. Set 'anndata=True' to get AnnData objects.")
        encode(data, autodetect=True)

    return data


def heart_failure(encoded: bool = False, columns_obs_only: dict[str, list[str]] | list[str] | None = None) -> AnnData:
    """Loads the heart failure dataset.

    More details: http://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
    Preprocessing: https://github.com/theislab/ehrapy-datasets/tree/main/heart_failure
    This dataset only contains numericals and therefore does not need any encoding.

    Args:
        encoded: Whether to return an already encoded object
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the MIMIC-II dataset

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.heart_failure(encoded=True)
    """
    adata = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/heart_failure.csv",
        download_dataset_name="heart_failure.csv",
        backup_url="https://figshare.com/ndownloader/files/33952934",
        columns_obs_only=columns_obs_only,
        index_column="patient_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

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

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.diabetes_130(encoded=True)
    """
    adata = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/diabetes_130.csv",
        download_dataset_name="diabetes_130.csv",
        backup_url="https://figshare.com/ndownloader/files/33950546",
        columns_obs_only=columns_obs_only,
        index_column="encounter_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def chronic_kidney_disease(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:  # pragma: no cover
    """Loads the Chronic Kidney Disease dataset

    More details: https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease
    Preprocessing: https://github.com/theislab/ehrapy-datasets/tree/main/chronic_kidney_disease/chronic_kidney_disease.ipynb

    Args:
        encoded: Whether to return an already encoded object
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Chronic Kidney Disease dataset

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.chronic_kidney_disease(encoded=True)
    """
    adata = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/chronic_kidney_disease_precessed.csv",
        download_dataset_name="chronic_kidney_disease.csv",
        backup_url="https://figshare.com/ndownloader/files/33989261",
        columns_obs_only=columns_obs_only,
        index_column="Patient_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def breast_tissue(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:
    """Loads the Breast Tissue Data Set

    More details: http://archive.ics.uci.edu/ml/datasets/Breast+Tissue
    Preprocessing: https://github.com/theislab/ehrapy-datasets/blob/main/breast_tissue/breast_tissue.ipynb

    Args:
        encoded: Whether to return an already encoded object
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Breast Tissue Data Set

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.breast_tissue(encoded=True)
    """
    adata = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/breast_tissue.csv",
        download_dataset_name="breast_tissue.csv",
        backup_url="https://figshare.com/ndownloader/files/34179264",
        columns_obs_only=columns_obs_only,
        index_column="patient_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def cervical_cancer_risk_factors(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:
    """Loads the Cervical cancer (Risk Factors) Data Set

    More details: http://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29
    Preprocessing: https://github.com/theislab/ehrapy-datasets/blob/main/cervical_cancer_risk_factors/cervical_cancer_risk_factors.ipynb

    Args:
        encoded: Whether to return an already encoded object
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Cervical cancer (Risk Factors) Data Set

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.cervical_cancer_risk_factors(encoded=True)
    """
    adata = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/cervical_cancer_risk_factors.csv",
        download_dataset_name="cervical_cancer_risk_factors.csv",
        backup_url="https://figshare.com/ndownloader/files/34179291",
        columns_obs_only=columns_obs_only,
        index_column="patient_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def dermatology(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:
    """Loads the Dermatology Data Set

    More details: http://archive.ics.uci.edu/ml/datasets/Dermatology
    Preprocessing: https://github.com/theislab/ehrapy-datasets/blob/main/dermatology/dermatology.ipynb

    Args:
        encoded: Whether to return an already encoded object
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Dermatology Data Set

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.dermatology(encoded=True)
    """
    adata = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/dermatology.csv",
        download_dataset_name="dermatology.csv",
        backup_url="https://figshare.com/ndownloader/files/34179300",
        columns_obs_only=columns_obs_only,
        index_column="patient_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def echocardiogram(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:
    """Loads the Echocardiogram Data Set

    More details: http://archive.ics.uci.edu/ml/datasets/Echocardiogram
    Preprocessing: https://github.com/theislab/ehrapy-datasets/blob/main/echocardiogram/echocardiogram.ipynb

    Args:
        encoded: Whether to return an already encoded object
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Echocardiogram Data Set

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.echocardiogram(encoded=True)
    """
    adata = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/echocardiogram.csv",
        download_dataset_name="echocardiogram.csv",
        backup_url="https://figshare.com/ndownloader/files/34179306",
        columns_obs_only=columns_obs_only,
        index_column="patient_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def hepatitis(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:
    """Loads the Hepatitis Data Set

    More details: http://archive.ics.uci.edu/ml/datasets/Hepatitis
    Preprocessing: https://github.com/theislab/ehrapy-datasets/blob/main/hepatitis/hepatitis.ipynb

    Args:
        encoded: Whether to return an already encoded object
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Hepatitis Data Set

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.hepatitis(encoded=True)
    """
    adata = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/hepatitis.csv",
        download_dataset_name="hepatitis.csv",
        backup_url="https://figshare.com/ndownloader/files/34179318",
        columns_obs_only=columns_obs_only,
        index_column="patient_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def statlog_heart(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:
    """Loads the Statlog (Heart) Data Set

    More details: http://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29
    Preprocessing: https://github.com/theislab/ehrapy-datasets/blob/main/statlog_heart/statlog_heart.ipynb

    Args:
        encoded: Whether to return an already encoded object
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Statlog (Heart) Data Set

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.statlog_heart(encoded=True)
    """
    adata = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/statlog_heart.csv",
        download_dataset_name="statlog_heart.csv",
        backup_url="https://figshare.com/ndownloader/files/34179327",
        columns_obs_only=columns_obs_only,
        index_column="patient_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def thyroid(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:
    """Loads the Thyroid Data Set

    More details: http://archive.ics.uci.edu/ml/datasets/Thyroid+Disease
    Preprocessing: https://github.com/theislab/ehrapy-datasets/blob/main/thyroid/thyroid.ipynb

    Args:
        encoded: Whether to return an already encoded object.
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Thyroid Data Set

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.thyroid(encoded=True)
    """
    adata: AnnData = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/thyroid.csv",
        download_dataset_name="thyroid.csv",
        backup_url="https://figshare.com/ndownloader/files/34179333",
        columns_obs_only=columns_obs_only,
        index_column="patient_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def breast_cancer_coimbra(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:
    """Loads the Breast Cancer Coimbra Data Set

    More details: http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra
    Preprocessing: https://github.com/theislab/ehrapy-datasets/blob/main/breast_cancer_coimbra/breast_cancer_coimbra.ipynb

    Args:
        encoded: Whether to return an already encoded object.
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Breast Cancer Coimbra Data Set

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.breast_cancer_coimbra(encoded=True)
    """
    adata: AnnData = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/breast_cancer_coimbra.csv",
        download_dataset_name="breast_cancer_coimbra.csv",
        backup_url="https://figshare.com/ndownloader/files/34439681",
        columns_obs_only=columns_obs_only,
        index_column="patient_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def parkinsons(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:
    """Loads the Parkinsons Data Set

    More details: http://archive.ics.uci.edu/ml/datasets/Parkinsons
    Preprocessing: https://github.com/theislab/ehrapy-datasets/blob/main/parkinsons/parkinsons.ipynb

    Args:
        encoded: Whether to return an already encoded object.
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Parkinsons Data Set

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.parkinsons(columns_obs_only=['name'], encoded=True)
    """
    adata: AnnData = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/parkinsons.csv",
        download_dataset_name="parkinsons.csv",
        backup_url="https://figshare.com/ndownloader/files/34439684",
        columns_obs_only=columns_obs_only,
        index_column="measurement_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def parkinsons_telemonitoring(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:
    """Loads the Parkinsons Telemonitoring Data Set

    More details: http://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring
    Preprocessing: https://github.com/theislab/ehrapy-datasets/blob/main/parkinsons_telemonitoring/parkinsons_telemonitoring.ipynb

    Args:
        encoded: Whether to return an already encoded object.
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Parkinsons Telemonitoring Data Set

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.parkinsons_telemonitoring(encoded=True)
    """
    adata: AnnData = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/parkinsons_telemonitoring.csv",
        download_dataset_name="parkinsons_telemonitoring.csv",
        backup_url="https://figshare.com/ndownloader/files/34439708",
        columns_obs_only=columns_obs_only,
        index_column="measurement_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def parkinsons_disease_classification(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:
    """Loads the Parkinson's Disease Classification Data Set

    More details: http://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification
    Preprocessing: https://github.com/theislab/ehrapy-datasets/blob/main/parkinson's_disease_classification/parkinson's_disease_classification.ipynb

    Args:
        encoded: Whether to return an already encoded object.
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Parkinson's Disease Classification Data Set

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.parkinsons_disease_classification(encoded=True)
    """
    adata: AnnData = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/parkinson's_disease_classification_prepared.csv",
        download_dataset_name="parkinson's_disease_classification_prepared.csv",
        backup_url="https://figshare.com/ndownloader/files/34439714",
        columns_obs_only=columns_obs_only,
        index_column="measurement_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def parkinson_dataset_with_replicated_acoustic_features(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:
    """Loads the Parkinson Dataset with replicated acoustic features Data Set

    More details: http://archive.ics.uci.edu/ml/datasets/Parkinson+Dataset+with+replicated+acoustic+features+
    Preprocessing: https://github.com/theislab/ehrapy-datasets/blob/main/parkinson_dataset_with_replicated_acoustic_features/parkinson_dataset_with_replicated_acoustic_features.ipynb

    Args:
        encoded: Whether to return an already encoded object.
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Parkinson Dataset with replicated acoustic features Data Set

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.parkinson_dataset_with_replicated_acoustic_features(columns_obs_only=['ID'], encoded=True)
    """
    adata: AnnData = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/parkinson_dataset_with_replicated_acoustic_features.csv",
        download_dataset_name="parkinson_dataset_with_replicated_acoustic_features.csv",
        backup_url="https://figshare.com/ndownloader/files/34439801",
        columns_obs_only=columns_obs_only,
        index_column="measurement_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def heart_disease(
    encoded: bool = False,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
) -> AnnData:
    """Loads the Heart Disease Data Set

    More details: http://archive.ics.uci.edu/ml/datasets/Heart+Disease
    Preprocessing: https://github.com/theislab/ehrapy-datasets/blob/main/heart_disease/heart_disease.ipynb

    Args:
        encoded: Whether to return an already encoded object.
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the Heart Disease Data Set

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.heart_disease(encoded=True)
    """
    adata: AnnData = read_csv(
        dataset_path=f"{ehrapy_settings.datasetdir}/processed_heart_disease.csv",
        download_dataset_name="processed_heart_disease.csv",
        backup_url="https://figshare.com/ndownloader/files/34906647",
        columns_obs_only=columns_obs_only,
        index_column="patient_id",
    )
    if encoded:
        return encode(adata, autodetect=True)

    return adata


def synthea_1k_sample(
    encoded: bool = False,
    columns_obs_only: list[str] | None = None,
) -> AnnData:
    """Loads the 1K Sample Synthetic Patient Records Data Set.

    More details: https://synthea.mitre.org/downloads
    Preprocessing: TODO: add preprocessing link

    Args:
        encoded: Whether to return an already encoded object.
        columns_obs_only: Columns to include in obs only and not X.

    Returns:
        :class:`~anndata.AnnData` object of the 1K Sample Synthetic Patient Records Data Set

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.synthea_1k_sample(encoded=True)
    """
    adata: AnnData = read_fhir(
        dataset_path=f"{ehrapy_settings.datasetdir}/synthea_sample",
        download_dataset_name="synthea_sample",
        backup_url="https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_fhir_dstu2_sep2019.zip",
        columns_obs_only=columns_obs_only,
        index_column="id",
        archive_format="zip",
    )

    if encoded:
        return encode(adata, autodetect=True)

    return adata
