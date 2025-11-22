[![Build](https://github.com/theislab/ehrapy/actions/workflows/build.yml/badge.svg)](https://github.com/theislab/ehrapy/actions/workflows/build.yml)
[![Codecov](https://codecov.io/gh/theislab/ehrapy/branch/master/graph/badge.svg)](https://codecov.io/gh/theislab/ehrapy)
[![License](https://img.shields.io/github/license/theislab/ehrapy)](https://opensource.org/licenses/Apache2.0)
[![PyPI](https://img.shields.io/pypi/v/ehrapy.svg)](https://pypi.org/project/ehrapy/)
[![Python Version](https://img.shields.io/pypi/pyversions/ehrapy)](https://pypi.org/project/ehrapy)
[![Read the Docs](https://img.shields.io/readthedocs/ehrapy/latest.svg?label=Read%20the%20Docs)](https://ehrapy.readthedocs.io/)
[![Test](https://github.com/theislab/ehrapy/actions/workflows/test.yml/badge.svg)](https://github.com/theislab/ehrapy/actions/workflows/test.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

<p align="center">
  <img src="https://user-images.githubusercontent.com/21954664/156930990-0d668468-0cd9-496e-995a-96d2c2407cf5.png" alt="ehrapy logo" width="25%">
</p>


# ehrapy overview

**ehrapy** is a modular open-source Python framework for exploratory analysis of heterogeneous epidemiological and EHR data.
It supports a full pipeline from data ingestion and quality control to advanced analyses such as clustering, survival, trajectory, causal inference, deep learning, and more.

<p align="center">
    <img src="https://github.com/user-attachments/assets/84fe403c-66de-4dd9-9265-b0d1739ce3cc" alt="fig1" width="100%">
</p>

## Documentation

Please read the [documentation](https://ehrapy.readthedocs.io/en/latest) for installation, tutorials, use cases, and more.

## Installation

You can install _ehrapy_ via [pip] from [PyPI]:

```console
$ pip install ehrapy
```

## API

Please have a look at the [API documentation](https://ehrapy.readthedocs.io/en/latest/api.html) and the [tutorials](https://ehrapy.readthedocs.io/en/latest/tutorials/index.html).

```python
import ehrapy as ep
```

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
[api]: https://ehrapy.readthedocs.io/en/latest/api.html
