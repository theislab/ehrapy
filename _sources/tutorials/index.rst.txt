:orphan:

Tutorials
==========

The easiest way to get familiar with ehrapy is to follow along with our tutorials.
Many are also designed to work seamlessly in Binder, a free cloud computing platform.

.. note:: For questions about the usage of ehrapy use `Github Discussions`_.

.. _Github Discussions: https://github.com/theislab/ehrapy/discussions


Quick start
-----------

.. nbgallery::

   notebooks/ehrapy_introduction
   notebooks/mimic_2_introduction
   notebooks/mimic_2_fate


Glossary
^^^^^^^^^

.. tabs::

    .. tab:: AnnData

        `AnnData <https://github.com/theislab/anndata>`_ is short for Annotated Data and is the primary datastructure that ehrapy uses.
        It is based on the principle of a single Numpy matrix X embraced by two Pandas Dataframes.
        All rows are called observations (in our case patients/patient visits or similar) and the columns
        are known as variables (any feature such as e.g. age, B12 level or similar).
        For a more in depth introduction please read the `AnnData paper <https://doi.org/10.1101/2021.12.16.473007>`_.

    .. tab:: scanpy

        The implementation of ehrapy is based on `scanpy <https://github.com/theislab/scanpy>`_, a framework to analyze single-cell sequencing data.
        ehrapy reuses the implemented algorithms in scanpy and wraps them for simple access.
        For a more in depth introduction please read the `Scanpy paper <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1382-0>`_.
