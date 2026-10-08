[![Build](https://github.com/theislab/ehrapy/actions/workflows/build.yml/badge.svg)](https://github.com/theislab/ehrapy/actions/workflows/build.yml)
[![Codecov](https://codecov.io/gh/theislab/ehrapy/branch/main/graph/badge.svg)](https://codecov.io/gh/theislab/ehrapy)
[![License](https://img.shields.io/github/license/theislab/ehrapy)](https://opensource.org/licenses/Apache2.0)
[![PyPI](https://img.shields.io/pypi/v/ehrapy.svg)](https://pypi.org/project/ehrapy/)
[![Python Version](https://img.shields.io/pypi/pyversions/ehrapy)](https://pypi.org/project/ehrapy)
[![Read the Docs](https://img.shields.io/readthedocs/ehrapy/latest.svg?label=Read%20the%20Docs)](https://ehrapy.readthedocs.io/)
[![Test](https://github.com/theislab/ehrapy/actions/workflows/test_cpu.yml/badge.svg)](https://github.com/theislab/ehrapy/actions/workflows/test_cpu.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

<p align="center">
  <img src="https://user-images.githubusercontent.com/21954664/156930990-0d668468-0cd9-496e-995a-96d2c2407cf5.png" alt="ehrapy logo" width="25%">
</p>

# ehrapy: electronic health record (EHR) analysis in Python

ehrapy is an open-source Python framework for exploratory and statistical analysis of electronic health records (EHR) and other clinical and epidemiological data.
It is for clinical researchers, epidemiologists and data scientists who want to go from raw patient data to quality-controlled cohorts, patient groups, trajectories, survival curves and treatment effect estimates in one reproducible workflow.
Where [PyHealth](https://github.com/sunlabuiuc/PyHealth) focuses on deep learning models for clinical prediction, the [OHDSI](https://www.ohdsi.org) tools on standardized observational studies of OMOP databases in R and SQL, and [MEDS](https://github.com/Medical-Event-Data-Standard/meds) on a common data standard for medical events, ehrapy covers exploratory and statistical analysis in Python.
It builds on [scverse](https://scverse.org) tools such as AnnData and scanpy and stores data in [ehrdata](https://github.com/theislab/ehrdata)'s `EHRData` object, which extends AnnData with a time axis.

## Features

- **Data access**: build `EHRData` objects from OMOP Common Data Model tables, CSV, h5ad, h5ed and zarr files with `ehrdata.io`, and load MIMIC-II, the MIMIC-IV OMOP demo, the PhysioNet 2012 and 2019 challenges, Synthea and other public datasets with `ehrdata.dt`.
- **Longitudinal data**: analyze static 2D and longitudinal 3D (patients × variables × time) data as numpy, sparse or dask arrays, aggregate time series with `ep.pp.summarize_measurements` and compare patients with dynamic time warping in `ep.pp.neighbors`.
- **Quality control**: missingness and summary statistics, laboratory reference ranges, Little's MCAR test and missing value plots.
- **Imputation and normalization**: explicit, mean or median, last observation carried forward, kNN and MissForest imputation, and scaling, log and power transforms.
- **Clustering and embeddings**: PCA, FAMD, UMAP, t-SNE and Leiden clustering, with feature ranking to characterize patient groups.
- **Trajectories**: diffusion pseudotime, PAGA and patterns across patients, variables and time with `ep.tl.ncp`.
- **Survival analysis**: Kaplan-Meier, Nelson-Aalen, Cox proportional hazards and accelerated failure time models, plus OLS and GLM regression.
- **Causal inference**: IPTW, g-computation, AIPW, propensity score matching and T-, S- and X-learners, with covariate balance and positivity diagnostics.
- **Fairness**: detect biases with respect to sensitive attributes such as sex or race with `ep.pp.detect_bias`.
- **Cohort reporting**: CONSORT-style cohort tracking and stratified Table One summaries.

<p align="center">
    <img src="https://github.com/user-attachments/assets/84fe403c-66de-4dd9-9265-b0d1739ce3cc" alt="ehrapy overview: data preparation, data preprocessing and knowledge inference" width="100%">
</p>

## Installation

You can install _ehrapy_ via [pip] from [PyPI]:

```console
$ pip install ehrapy
```

Optional extras enable dask-backed out-of-core arrays (`ehrapy[dask]`), Leiden clustering (`ehrapy[leiden]`), and GPU acceleration through rapids-singlecell (`ehrapy[rapids12]` or `ehrapy[rapids13]`).

## Quickstart

Cluster 1,776 intensive care patients from MIMIC-II and color them by in-hospital mortality, after `pip install "ehrapy[leiden]"`:

```python
import ehrdata as ed
import ehrapy as ep

edata = ed.dt.mimic_2(columns_obs_only=["service_unit", "day_icu_intime", "hosp_exp_flg"])
ed.infer_feature_types(edata, binary_as="numeric")
ep.pp.qc_metrics(edata)
ep.pp.knn_impute(edata)
ep.pp.scale_norm(edata)
ep.pp.pca(edata)
ep.pp.neighbors(edata)
ep.tl.leiden(edata)
ep.tl.umap(edata)
ep.pl.umap(edata, color=["leiden", "service_unit", "hosp_exp_flg"])
```

## Documentation

The [documentation](https://ehrapy.readthedocs.io) has the [tutorials](https://ehrapy.readthedocs.io/en/stable/tutorials/index.html) and the [API reference](https://ehrapy.readthedocs.io/en/stable/api.html).

## Citation

 <p align="center">
  <a href="https://www.nature.com/articles/s41591-024-03214-0">
    <img src="https://github.com/user-attachments/assets/c3f7e79d-1633-4767-9dda-e94262279685" alt="fig2" width="50%">
  </a>
</p>

Read more about ehrapy in the [associated publication](https://doi.org/10.1038/s41591-024-03214-0).

```bibtex
@article{Heumos2024,
  author = {Heumos, Lukas and Ehmele, Philipp and Treis, Tim and Upmeier zu Belzen, Julius and Roellin, Eljas and May, Lilly and Namsaraeva, Altana and Horlava, Nastassya and Shitov, Vladimir A. and Zhang, Xinyue and Zappia, Luke and Knoll, Rainer and Lang, Niklas J. and Hetzel, Leon and Virshup, Isaac and Sikkema, Lisa and Curion, Fabiola and Eils, Roland and Schiller, Herbert B. and Hilgendorff, Anne and Theis, Fabian J.},
  year = {2024},
  month = {11},
  day = {01},
  title = {An open-source framework for end-to-end analysis of electronic health record data},
  journal = {Nature Medicine},
  volume = {30},
  number = {11},
  pages = {3369--3380},
  issn = {1546-170X},
  doi = {10.1038/s41591-024-03214-0},
  url = {https://doi.org/10.1038/s41591-024-03214-0}
}
```

[pip]: https://pip.pypa.io/
[pypi]: https://pypi.org/
