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
It reads static and longitudinal patient data from OMOP databases, public datasets such as MIMIC and PhysioNet, or your own tables.
It is for clinical researchers, epidemiologists and data scientists who want to go from raw patient data to quality-controlled cohorts, patient groups, trajectories, statistical tests, survival curves and treatment effect estimates in one reproducible workflow.
With `ep.ml`, it also trains and evaluates prediction models on the same data, from gradient boosting to recurrent and transformer networks, with patient-level splits, calibration and subgroup metrics.

<p align="center">
    <img src="https://github.com/user-attachments/assets/84fe403c-66de-4dd9-9265-b0d1739ce3cc" alt="ehrapy overview: data preparation, data preprocessing and knowledge inference" width="100%">
</p>

## Installation

You can install _ehrapy_ via [pip] from [PyPI]:

```console
$ pip install ehrapy
```

Optional extras enable dask-backed out-of-core arrays (`ehrapy[dask]`), Leiden clustering (`ehrapy[leiden]`), deep learning prediction models (`ehrapy[ml]`), and GPU acceleration through rapids-singlecell (`ehrapy[rapids12]` or `ehrapy[rapids13]`).

## Quickstart

Cluster 11,988 intensive care stays from PhysioNet 2012, each with 37 measurements over 48 hours, and test which measurements differ between patients who died and survived, after `pip install "ehrapy[leiden]"`:

```python
import ehrdata as ed
import ehrapy as ep

edata = ed.dt.physionet2012()  # 11,988 ICU stays × 37 measurements × 48 hours
ed.infer_feature_types(edata)
ep.pp.locf_impute(edata)
ep.pp.scale_norm(edata)

summary = ep.pp.summarize_measurements(edata, statistics=["mean", "min", "max"])
summary.obs["outcome"] = summary.obs["In-hospital_death"].map({0: "survived", 1: "died"}).astype("category")
ep.pp.pca(summary)
ep.pp.neighbors(summary)
ep.tl.leiden(summary, resolution=0.3)
ep.tl.umap(summary)
ep.pl.umap(summary, color=["leiden", "outcome"])

ep.tl.rank_features_groups(summary, groupby="outcome", groups=["died"], reference="survived")
ep.get.rank_features_groups_df(summary, group="died")[["names", "scores", "pvals_adj"]].head()
```

<p align="center">
    <img src="https://raw.githubusercontent.com/theislab/ehrapy/main/docs/_static/readme_quickstart.png" alt="UMAP of PhysioNet 2012 intensive care stays colored by Leiden cluster and in-hospital death" width="100%">
</p>

```
        names     scores      pvals_adj
0    GCS_mean -28.216665  1.869927e-146
1     GCS_max -22.782927  1.980853e-100
2    BUN_mean  18.654604   3.497719e-70
3     BUN_min  18.486603   2.324785e-69
4  Urine_mean -16.941847   1.399260e-59
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
