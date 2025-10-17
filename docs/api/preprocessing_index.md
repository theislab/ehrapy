# Preprocessing

Any transformation of the data matrix that is not a tool.
Preprocessing steps usually perform a basic transformation on the data array in `.X` or `.layers`.

```{eval-rst}
.. module:: ehrapy
    :no-index:
```

## Basic preprocessing

```{eval-rst}
.. autosummary::
    :toctree: preprocessing
    :nosignatures:

    preprocessing.encode
    preprocessing.pca
    preprocessing.regress_out
    preprocessing.sample
    preprocessing.highly_variable_features
    preprocessing.winsorize
    preprocessing.clip_quantile
    preprocessing.summarize_measurements
```
## Filtering

```{eval-rst}
.. autosummary::
    :toctree: preprocessing
    :nosignatures:

    preprocessing.filter_features
    preprocessing.filter_observations
```


## Quality control

```{eval-rst}
.. autosummary::
    :toctree: preprocessing
    :nosignatures:

    preprocessing.qc_metrics
    preprocessing.qc_lab_measurements
    preprocessing.mcar_test
    preprocessing.detect_bias
```

## Imputation

```{eval-rst}
.. autosummary::
    :toctree: preprocessing
    :nosignatures:

    preprocessing.explicit_impute
    preprocessing.simple_impute
    preprocessing.knn_impute
    preprocessing.miss_forest_impute
    preprocessing.mice_forest_impute
```

## Normalization

```{eval-rst}
.. autosummary::
    :toctree: preprocessing
    :nosignatures:

    preprocessing.log_norm
    preprocessing.maxabs_norm
    preprocessing.minmax_norm
    preprocessing.power_norm
    preprocessing.quantile_norm
    preprocessing.robust_scale_norm
    preprocessing.scale_norm
    preprocessing.offset_negative_values
```

## Dataset Shift Correction

Partially overlaps with dataset integration. Note that a simple batch correction method is available via `pp.regress_out()`.

```{eval-rst}
.. autosummary::
    :toctree: preprocessing
    :nosignatures:

    preprocessing.combat
```

## Neighbors

```{eval-rst}
.. autosummary::
    :toctree: preprocessing
    :nosignatures:

    preprocessing.neighbors
```
