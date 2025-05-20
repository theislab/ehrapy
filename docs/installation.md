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

If you run into "RuntimeError: CMake must be installed to build qdldl" ensure that you have CMake installed to build lightgbm.
Run `conda install -c anaconda cmake` and `conda install -c conda-forge lightgbm` to do so.

### Optional dependencies

#### causal & dowhy

To run causal inference with ehrapy, install the `causal` extra:

```console
pip install ehrapy[causal]
```

## From sources

The sources for ehrapy can be downloaded from the [Github repo].

You can either clone the public repository:

```console
git clone git://github.com/theislab/ehrapy
```

Or download the [tarball]:

```console
curl -OJL https://github.com/theislab/ehrapy/tarball/master
```

[github repo]: https://github.com/theislab/ehrapy
[pip]: https://pip.pypa.io
[python installation guide]: http://docs.python-guide.org/en/latest/starting/installation/
[tarball]: https://github.com/theislab/ehrapy/tarball/master
