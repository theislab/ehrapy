Contributor Guide
=================

Thank you for your interest in improving this project.
This project is open-source under the `Apache2.0 license`_ and
highly welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- `Source Code`_
- `Documentation`_
- `Issue Tracker`_
- `Code of Conduct`_

.. _Apache2.0 license: https://opensource.org/licenses/Apache2.0
.. _Source Code: https://github.com/theislab/ehrapy
.. _Documentation: https://ehrapy.readthedocs.io/
.. _Issue Tracker: https://github.com/theislab/ehrapy/issues

How to report a bug
-------------------

Report bugs on the `Issue Tracker`_.


How to request a feature
------------------------

Request features on the `Issue Tracker`_.

Getting the code
----------------

ehrapy uses submodules for the tutorials. Hence, the project must be cloned as:

.. code:: console

    $ git clone --recurse-submodules --remote-submodules https://github.com/theislab/ehrapy

This will automatically also clone and update the submodules.

How to set up your development environment
------------------------------------------

You need Python 3.8+ and the following tools:

- Poetry_
- Nox_
- nox-poetry_

You can install them with:

.. code:: console

    $ pip install poetry nox nox-poetry

Install the package with development requirements:

.. code:: console

   $ make install

You can now run an interactive Python session,
or the command-line interface:

.. code:: console

   $ poetry run python
   $ poetry run ehrapy

.. _Poetry: https://python-poetry.org/
.. _Nox: https://nox.thea.codes/
.. _nox-poetry: https://nox-poetry.readthedocs.io/

How to test the project
-----------------------

Run the full test suite:

.. code:: console

   $ nox

List the available Nox sessions:

.. code:: console

   $ nox --list-sessions

You can also run a specific Nox session.
For example, invoke the unit test suite like this:

.. code:: console

   $ nox --session=tests

Unit tests are located in the ``tests`` directory,
and are written using the pytest_ testing framework.

.. _pytest: https://pytest.readthedocs.io/

How to build and view the documentation
---------------------------------------

This project uses Sphinx_ together with several extensions to build the documentation.
It further requires Pandoc_ to translate various formats.

To install all required dependencies for the documentation run:

.. code:: console

    $ pip install -r docs/requirements.txt

Please note that ehrapy itself must also be installed. To build the documentation run:

.. code:: console

    $ make html

from inside the docs folder. The generated static HTML files can be found in the `_build/html` folder.
Simply open them with your favorite browser.

.. _sphinx: https://www.sphinx-doc.org/en/master/
.. _pandoc: https://pandoc.org/

How to submit changes
---------------------

Open a `pull request`_ to submit changes to this project against the ``development`` branch.

Your pull request needs to meet the following guidelines for acceptance:

- The Nox test suite must pass without errors and warnings.
- Include unit tests. This project maintains a high code coverage.
- If your changes add functionality, update the documentation accordingly.

To run linting and code formatting checks before committing your change, you can install pre-commit as a Git hook by running the following command:

.. code:: console

   $ nox --session=pre-commit -- install

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

.. _pull request: https://github.com/theislab/ehrapy/pulls
.. _Code of Conduct: CODE_OF_CONDUCT.rst
