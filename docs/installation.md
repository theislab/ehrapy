```{highlight} shell

```

# Installation

## Stable release

To install ehrapy, run this command in your terminal:

```console
$ pip install ehrapy
```

This is the preferred method to install ehrapy, as it will always install the most recent stable release.

If you don't have [pip] installed, this [Python installation guide] can guide you through the process.

If you run into "RuntimeError: CMake must be installed to build qdldl" ensure that you have CMake installed to build lightgbm.
Run `conda install -c anaconda cmake` and `conda install -c conda-forge lightgbm` to do so.

If you intend to run MedCAT you have to install a language model like:

```console
$ python -m spacy download en_core_web_sm
```

## From sources

The sources for ehrapy can be downloaded from the [Github repo].
Please note that you require [poetry] to be installed.

You can either clone the public repository:

```console
$ git clone git://github.com/theislab/ehrapy
```

Or download the [tarball]:

```console
$ curl -OJL https://github.com/theislab/ehrapy/tarball/master
```

Once you have a copy of the source, you can install it with:

```console
$ make install
```

To install MedCAT/Spacy language models you can run the installation with extra dependency groups like:

```console
$ poetry install -E en_core_web_md
```

## MedCAT/Spacy language models

If you want to run and use medcat with ehrapy, you first have to install medcat:

```console
$ poetry install -E medcat
```

Available language models are

-   en_core_web_md (python -m spacy download en_core_web_md)
-   en-core-sci-sm (pip install <https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz>)
-   en-core-sci-md (pip install <https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_md-0.4.0.tar.gz>)
-   en-core-sci-lg (pip install <https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz>)

[github repo]: https://github.com/theislab/ehrapy
[pip]: https://pip.pypa.io
[poetry]: https://python-poetry.org/
[python installation guide]: http://docs.python-guide.org/en/latest/starting/installation/
[tarball]: https://github.com/theislab/ehrapy/tarball/master
