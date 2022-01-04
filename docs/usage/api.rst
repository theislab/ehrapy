===
API
===

Import the ehrapy API as follows:

.. code:: python

   import ehrapy.api as ep

You can then access the respective modules like:

.. code:: python

   eh.pl.cool_fancy_plot()


.. currentmodule:: ehrapy.api

Reading and writing
~~~~~~~~~~~~~~~~~~~~

.. module:: ehrapy.api

.. autosummary::
    :toctree: io

    io.read
    io.write

Data
~~~~~

.. autosummary::
    :toctree: data

    data.mimic_2
    data.mimic_3_demo

Preprocessing
~~~~~~~~~~~~~

Any transformation of the data matrix that is not a tool.
Other than tools, preprocessing steps usually don’t return an easily interpretable annotation, but perform a basic transformation on the data matrix.

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

Quality control
+++++++++++++++

.. autosummary::
    :toctree: preprocessing

    preprocessing.calculate_qc_metrics

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

.. autosummary::
    :toctree: tools

    tools.leiden
    tools.louvain
    tools.dendrogram
    tools.dpt
    tools.paga

Group comparison
++++++++++++++++

.. autosummary::
    :toctree: tools

    tools.rank_features_groups
    tools.filter_rank_features_groups
    tools.marker_feature_overlap

Dataset integration
+++++++++++++++++++

.. currentmodule:: ehrapy.api

.. autosummary::
    :toctree: tools

    tools.ingest

Translators
+++++++++++

.. autosummary::
    :toctree: tools

    tools.DeepL

Plotting
~~~~~~~~

The plotting module ehrapy.pl largely parallels the tl.* and a few of the pp.* functions.
For most tools and for some preprocessing functions, you’ll find a plotting function with the same name.

Generic
+++++++

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

.. autosummary::
    :toctree: plot

    plot.pca
    plot.pca_loadings
    plot.pca_variance_ratio
    plot.pca_overview

Embeddings
++++++++++

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

.. autosummary::
    :toctree: plot

    plot.dpt_groups_pseudotime
    plot.dpt_timeseries
    plot.paga
    plot.paga_path
    plot.paga_compare

Group comparison
++++++++++++++++

.. autosummary::
    :toctree: plot

    plot.rank_features_groups
    plot.rank_features_groups_violin
    plot.rank_features_groups_stacked_violin
    plot.rank_features_groups_heatmap
    plot.rank_features_groups_dotplot
    plot.rank_features_groups_matrixplot
    plot.rank_features_groups_tracksplot

Settings
~~~~~~~~

A convenience object for setting some default :obj:`matplotlib.rcParams` and a
high-resolution jupyter display backend useful for use in notebooks.

An instance of the :class:`~scanpy._settings.ScanpyConfig` is available as `ehrapy.settings` and allows configuring ehrapy.

.. code-block:: python

    import ehrapy.api as ep
    ep.settings.set_figure_params(dpi=150)

Please refer to the `Scanpy settings documentation <https://scanpy.readthedocs.io/en/stable/api.html#settings>`_
for configuration options. Ehrapy will adapt these in the future and update the documentation.

Dependency Versions
~~~~~~~~~~~~~~~~~~~

ehrapy is complex software with many dependencies. To ensure a consistent runtime environment you should save
the tool versions of a conducted analysis. This comes in handy when trying to diagnose issues and to reproduce results.

Call the function via:

.. code-block:: python

    ep.print_versions()
