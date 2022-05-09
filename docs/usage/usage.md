# Usage

Import the ehrapy API as follows:

```python
import ehrapy as ep
```

You can then access the respective modules like:

```python
ep.pl.cool_fancy_plot()
```

```{eval-rst}
.. currentmodule:: ehrapy
```

## Reading and writing

```{eval-rst}
.. module:: ehrapy
```

```{eval-rst}
.. autosummary::
    :toctree: io
    :nosignatures:

    io.read_csv
    io.read_h5ad
    io.read_pdf
    io.write
```

## Data

```{eval-rst}
.. autosummary::
    :toctree: data
    :nosignatures:

    data.mimic_2
    data.mimic_3_demo
    data.diabetes_130
    data.heart_failure
    data.chronic_kidney_disease
    data.breast_tissue
    data.cervical_cancer_risk_factors
    data.dermatology
    data.echocardiogram
    data.heart_disease
    data.hepatitis
    data.statlog_heart
    data.thyroid
    data.breast_cancer_coimbra
    data.parkinson_dataset_with_replicated_acoustic_features
    data.parkinsons
    data.parkinsons_disease_classification
    data.parkinsons_telemonitoring
```

## Preprocessing

Any transformation of the data matrix that is not a tool.
Other than tools, preprocessing steps usually don’t return an easily interpretable annotation, but perform a basic transformation on the data matrix.

### Basic preprocessing

```{eval-rst}
.. autosummary::
    :toctree: preprocessing
    :nosignatures:

    preprocessing.pca
    preprocessing.regress_out
    preprocessing.subsample
    preprocessing.highly_variable_features
```

### Quality control

```{eval-rst}
.. autosummary::
    :toctree: preprocessing
    :nosignatures:

    preprocessing.qc_metrics
    preprocessing.qc_lab_measurements
```

### Imputation

```{eval-rst}
.. autosummary::
    :toctree: imputation
    :nosignatures:

    preprocessing.explicit_impute
    preprocessing.simple_impute
    preprocessing.knn_impute
    preprocessing.miss_forest_impute
    preprocessing.soft_impute
    preprocessing.iterative_svd_impute
    preprocessing.matrix_factorization_impute
    preprocessing.nuclear_norm_minimization_impute
    preprocessing.miceforest_impute
```

### Encoding

```{eval-rst}
.. autosummary::
    :toctree: preprocessing
    :nosignatures:

    preprocessing.encode
    preprocessing.undo_encoding
```

### Normalization

```{eval-rst}
.. autosummary::
    :toctree: preprocessing
    :nosignatures:

    preprocessing.norm_log
    preprocessing.norm_maxabs
    preprocessing.norm_minmax
    preprocessing.norm_power
    preprocessing.norm_quantile
    preprocessing.norm_robust_scale
    preprocessing.norm_scale
    preprocessing.norm_sqrt
```

### Dataset Shift Correction

Partially overlaps with dataset integration. Note that a simple batch correction method is available via `pp.regress_out()`.

```{eval-rst}
.. autosummary::
    :toctree: preprocessing
    :nosignatures:

    preprocessing.combat
```

### Neighbors

```{eval-rst}
.. autosummary::
    :toctree: preprocessing
    :nosignatures:

    preprocessing.neighbors
```

## Tools

Any transformation of the data matrix that is not preprocessing.
In contrast to a preprocessing function, a tool usually adds an easily interpretable annotation to the data matrix, which can then be visualized with a corresponding plotting function.

### Embeddings

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.pca
    tools.tsne
    tools.umap
    tools.draw_graph
    tools.diffmap
    tools.embedding_density
```

### Clustering and trajectory inference

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.leiden
    tools.louvain
    tools.dendrogram
    tools.dpt
    tools.paga
```

### Group comparison

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.rank_features_groups
    tools.filter_rank_features_groups
    tools.marker_feature_overlap
```

### Dataset integration

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.ingest
```

### Natural language processing

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.Translator
    tools.HPOMapper
    tools.MedCAT
```

### Data Analysis (Linear Regression, Logistic Regression)

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.ols
    tools.glm
```

## Plotting

The plotting module ehrapy.pl largely parallels the tl.\* and a few of the pp.\* functions.
For most tools and for some preprocessing functions, you will find a plotting function with the same name.

### Generic

```{eval-rst}
.. autosummary::
    :toctree: plot
    :nosignatures:

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
```

### Quality Control

```{eval-rst}
.. autosummary::
    :toctree: plot
    :nosignatures:

    plot.qc_metrics
```

### Classes

Please refer to [Scanpy's plotting classes documentation](https://scanpy.readthedocs.io/en/stable/api.html#classes).

### Tools

Methods that extract and visualize tool-specific annotation in an AnnData object. For any method in module `tl`, there is a method with the same name in `pl`.

```{eval-rst}
.. autosummary::
    :toctree: plot
    :nosignatures:

    plot.pca
    plot.pca_loadings
    plot.pca_variance_ratio
    plot.pca_overview
```

### Embeddings

```{eval-rst}
.. autosummary::
    :toctree: plot
    :nosignatures:

    plot.tsne
    plot.umap
    plot.diffmap
    plot.draw_graph
    plot.spatial
    plot.embedding
    plot.embedding_density
```

### Branching trajectories and pseudotime, clustering

Visualize clusters using one of the embedding methods passing color='leiden'.

```{eval-rst}
.. autosummary::
    :toctree: plot
    :nosignatures:

    plot.dpt_groups_pseudotime
    plot.dpt_timeseries
    plot.paga
    plot.paga_path
    plot.paga_compare
```

### Group comparison

```{eval-rst}
.. autosummary::
    :toctree: plot
    :nosignatures:

    plot.rank_features_groups
    plot.rank_features_groups_violin
    plot.rank_features_groups_stacked_violin
    plot.rank_features_groups_heatmap
    plot.rank_features_groups_dotplot
    plot.rank_features_groups_matrixplot
    plot.rank_features_groups_tracksplot
```

### Data Analysis (Linear Regression, Survival Analysis)

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    plot.ols
    plot.kmf
```

## AnnData object transformation

The ehrapy API exposes functions to transform a pandas dataframe into an AnnData object
and vice versa.

```{eval-rst}
.. autosummary::
    :toctree: ad
    :nosignatures:

    anndata.df_to_anndata
    anndata.anndata_to_df
    anndata.move_to_obs
    anndata.move_to_x
    anndata.type_overview

```

## Settings

A convenience object for setting some default {obj}`matplotlib.rcParams` and a
high-resolution jupyter display backend useful for use in notebooks.

An instance of the {class}`~scanpy._settings.ScanpyConfig` is available as `ehrapy.settings` and allows configuring ehrapy.

```python
import ehrapy as ep
ep.settings.set_figure_params(dpi=150)
```

Please refer to the [Scanpy settings documentation](https://scanpy.readthedocs.io/en/stable/api.html#settings)
for configuration options. Ehrapy will adapt these in the future and update the documentation.

## Dependency Versions

ehrapy is complex software with many dependencies. To ensure a consistent runtime environment you should save
the tool versions of a conducted analysis. This comes in handy when trying to diagnose issues and to reproduce results.

Call the function via:

```python
ep.print_versions()
```
