```{highlight} shell

```

# Installation

## Stable release

To install ehrapy, run this command in your terminal:

```console
pip install ehrapy
```

This is the preferred method to install ehrapy, as it will always install the most recent stable release.

If you don't have [pip] installed, this [Python installation guide] can guide you through the process.

### Optional dependencies

ehrapy keeps optional functionality in extras:

| Extra | Enables |
| --- | --- |
| `dask` | Out-of-core and lazy computation on {class}`dask.array.Array` data |
| `leiden` | {func}`ehrapy.tools.leiden` clustering through `igraph` (GPL licensed) |
| `rapids12`, `rapids13` | GPU acceleration through rapids-singlecell for CUDA 12 or 13 |

Install one or several extras at once:

```console
pip install "ehrapy[dask,leiden]"
```

## From sources

To install the latest development version directly from [GitHub]:

```console
pip install git+https://github.com/theislab/ehrapy
```

To work on ehrapy itself, clone the repository and follow the {doc}`contributing guide <contributing>`:

```console
git clone https://github.com/theislab/ehrapy
```

[github]: https://github.com/theislab/ehrapy
[pip]: https://pip.pypa.io
[python installation guide]: https://docs.python-guide.org/starting/installation/
