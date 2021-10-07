ehrapy
===========================

|PyPI| |Python Version| |License| |Read the Docs| |Build| |Tests| |Codecov| |pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/ehrapy.svg
   :target: https://pypi.org/project/ehrapy/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/ehrapy
   :target: https://pypi.org/project/ehrapy
   :alt: Python Version
.. |License| image:: https://img.shields.io/github/license/theislab/ehrapy
   :target: https://opensource.org/licenses/Apache2.0
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/ehrapy/latest.svg?label=Read%20the%20Docs
   :target: https://ehrapy.readthedocs.io/
   :alt: Read the documentation at https://ehrapy.readthedocs.io/
.. |Build| image:: https://github.com/theislab/ehrapy/workflows/Build%20ehrapy%20Package/badge.svg
   :target: https://github.com/theislab/ehrapy/actions?workflow=Package
   :alt: Build Package Status
.. |Tests| image:: https://github.com/theislab/ehrapy/workflows/Run%20ehrapy%20Tests/badge.svg
   :target: https://github.com/theislab/ehrapy/actions?workflow=Tests
   :alt: Run Tests Status
.. |Codecov| image:: https://codecov.io/gh/theislab/ehrapy/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/theislab/ehrapy
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black


Features
--------

* Exploratory analysis of Electronic Health Records
* Quality control & preprocessing
* Clustering & trajectory inference
* Visualization & Exploration


Installation
------------

You can install *ehrapy* via pip_ from PyPI_:

.. code:: console

   $ pip install ehrapy


Usage
-----

Please see the `Usage documentation <Usage_>`_ for details.

.. code:: python

   import ehra.api as ep


Credits
-------

This package was created with cookietemple_ using Cookiecutter_ based on Hypermodern_Python_Cookiecutter_.

.. _cookietemple: https://cookietemple.com
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _PyPI: https://pypi.org/
.. _Hypermodern_Python_Cookiecutter: https://github.com/cjolowicz/cookiecutter-hypermodern-python
.. _pip: https://pip.pypa.io/
.. _Usage: https://ehrapy.readthedocs.io/en/latest/usage.html
