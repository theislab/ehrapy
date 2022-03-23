# Contributor Guide

Thank you for your interest in improving this project.
This project is open-source under the [Apache2.0 license] and
highly welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

-   [Source Code]
-   [Documentation]
-   [Issue Tracker]
-   [Code of Conduct]

## How to report a bug

Report bugs on the [Issue Tracker].

## How to request a feature

Request features on the [Issue Tracker].

## Getting the code

ehrapy uses submodules for the tutorials. Hence, the project must be cloned as:

```console
$ git clone --recurse-submodules --remote-submodules https://github.com/theislab/ehrapy
```

This will automatically also clone and update the submodules.

## How to set up your development environment

You need Python 3.8+ and the following tools:

-   [Poetry]
-   [Nox]
-   [nox-poetry]

You can install them with:

```console
$ pip install poetry nox nox-poetry
```

Install the package with development requirements:

```console
$ make install
```

You can now run an interactive Python session,
or the command-line interface:

```console
$ poetry run python
$ poetry run ehrapy
```

## How to test the project

Run the full test suite:

```console
$ nox
```

List the available Nox sessions:

```console
$ nox --list-sessions
```

You can also run a specific Nox session.
For example, invoke the unit test suite like this:

```console
$ nox --session=tests
```

Unit tests are located in the `tests` directory,
and are written using the [pytest] testing framework.

## How to build and view the documentation

This project uses [Sphinx] together with several extensions to build the documentation.
It further requires [Pandoc] to translate various formats.

To install all required dependencies for the documentation run:

```console
$ pip install -r docs/requirements.txt
```

Please note that ehrapy itself must also be installed. To build the documentation run:

```console
$ make html
```

from inside the docs folder. The generated static HTML files can be found in the `_build/html` folder.
Simply open them with your favorite browser.

## How to submit changes

Open a [pull request] to submit changes to this project against the `development` branch.

Your pull request needs to meet the following guidelines for acceptance:

-   The Nox test suite must pass without errors and warnings.
-   Include unit tests. This project maintains a high code coverage.
-   If your changes add functionality, update the documentation accordingly.

To run linting and code formatting checks before committing your change, you can install pre-commit as a Git hook by running the following command:

```console
$ nox --session=pre-commit -- install
```

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

[apache2.0 license]: https://opensource.org/licenses/Apache2.0
[code of conduct]: CODE_OF_CONDUCT.md
[documentation]: https://ehrapy.readthedocs.io/
[issue tracker]: https://github.com/theislab/ehrapy/issues
[nox]: https://nox.thea.codes/
[nox-poetry]: https://nox-poetry.readthedocs.io/
[pandoc]: https://pandoc.org/
[poetry]: https://python-poetry.org/
[pull request]: https://github.com/theislab/ehrapy/pulls
[pytest]: https://pytest.readthedocs.io/
[source code]: https://github.com/theislab/ehrapy
[sphinx]: https://www.sphinx-doc.org/en/master/
