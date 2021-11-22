Usage
=====

API
---

Import the ehrapy API as follows:

.. code:: python

   import ehrapy.api as ep

You can then access the respective modules like:

.. code:: python

   eh.pl.cool_fancy_plot()

.. contents::
    :local:
    :backlinks: none

Reading and writing
~~~~~~~~~~~~~~~~~~~~

.. module:: ehrapy.api
.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: io

    io.read

    io.write


Data
~~~~~

.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: data

    data.mimic_2

    data.mimic_3

Preprocessing
~~~~~~~~~~~~~

Any transformation of the data matrix that is not a tool.
Other than tools, preprocessing steps usually don’t return an easily interpretable annotation, but perform a basic transformation on the data matrix.

.. currentmodule:: ehrapy.api

Basic preprocessing
+++++++++++++++++++

.. autosummary::
    :toctree: preprocessing

    preprocessing.replace_explicit
    preprocessing.log1p
    preprocessing.pca
    preprocessing.normalize_total
    preprocessing.regress_out
    preprocessing.scale
    preprocessing.subsample

Encoding
++++++++

.. autosummary::
    :toctree: preprocessing

    preprocessing.encode
    preprocessing.undo_encoding
    preprocessing.type_overview

Dataset Shift Correction
++++++++++++++++++++++++

Partially overlaps with dataset integration. Note that a simple batch correction method is available via `pp.regress_out()`.

.. autosummary::
    :toctree: preprocessing

    preprocessing.combat

Neighbors
+++++++++

.. autosummary::
    :toctree: preprocessing

    preprocessing.neighbors

Tools
~~~~~

Any transformation of the data matrix that is not preprocessing.
In contrast to a preprocessing function, a tool usually adds an easily interpretable annotation to the data matrix, which can then be visualized with a corresponding plotting function.

Embeddings
++++++++++

.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: tools

    tools.pca
    tools.tsne
    tools.umap
    tools.draw_graph
    tools.diffmap
    tools.embedding_density

Clustering and trajectory inference
+++++++++++++++++++++++++++++++++++

.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: tools

    tools.leiden
    tools.louvain
    tools.dendrogram
    tools.dpt
    tools.paga

Dataset integration
+++++++++++++++++++

.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: tools

    tools.ingest

Translators
+++++++++++

.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: tools

    tools.DeepL

Plotting
~~~~~~~~

The plotting module scanpy.pl largely parallels the tl.* and a few of the pp.* functions.
For most tools and for some preprocessing functions, you’ll find a plotting function with the same name.

Generic
+++++++

.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: plot

    plot.scatter
    plot.heatmap
    plot.dotplot
    plot.tracksplot
    plot.violin
    plot.stacked_violin
    plot.matrixplot
    plot.clustermap
    plot.ranking
    plot.dendrogram

Classes
+++++++

Please refer to `Scanpy's plotting classes documentation <https://scanpy.readthedocs.io/en/stable/api.html#classes>`_.

Preprocessing
+++++++++++++

Not available at the moment.

Tools
+++++

Methods that extract and visualize tool-specific annotation in an AnnData object. For any method in module `tl`, there is a method with the same name in `pl`.

.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: plot

    plot.pca
    plot.pca_loadings
    plot.pca_variance_ratio
    plot.pca_overview

Embeddings
++++++++++

.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: plot

    plot.tsne
    plot.umap
    plot.diffmap
    plot.draw_graph
    plot.spatial
    plot.embedding
    plot.embedding_density

Branching trajectories and pseudotime, clustering
+++++++++++++++++++++++++++++++++++++++++++++++++

Visualize clusters using one of the embedding methods passing color='leiden'.

.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: plot

    plot.dpt_groups_pseudotime
    plot.dpt_timeseries
    plot.paga
    plot.paga_path
    plot.paga_compare

Settings
~~~~~~~~

A convenience object for setting some default :obj:`matplotlib.rcParams` and a
high-resolution jupyter display backend useful for use in notebooks.

An instance of the :class:`~scanpy._settings.ScanpyConfig` is available as `ehrapy.settings` and allows configuring ehrapy.

Please refer to the `Scanpy settings documentation <https://scanpy.readthedocs.io/en/stable/api.html#settings>`_
for configuration options. Ehrapy will adapt these in the future and update the documentation.

Dependency Versions
~~~~~~~~~~~~~~~~~~~

It is possible to get the versions of all dependencies in the current runtime environment.
This comes in handy when trying to diagnose issues.

Call the function via:

.. code-block:: python

    ep.print_versions()

Command-line interface
-----------------------

.. click:: ehrapy.__main__:ehrapy_cli
   :prog: ehrapy
   :nested: full
