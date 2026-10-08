# Data dimensions and array types

ehrapy operates on {class}`~ehrdata.EHRData` objects.
Their `X` and `layers` hold either static data of shape `(n_obs, n_vars)` or longitudinal data of shape `(n_obs, n_vars, n_t)`, where the third axis is described by `edata.tem`.

Every array can be a {class}`numpy.ndarray`, a scipy sparse matrix or array in CSR or CSC format, or a {class}`dask.array.Array` with numpy chunks.

## How functions treat array types

Functions follow the same rules for every array type:

- Dask arrays stay lazy: functions that return or store arrays never compute them, and functions that store small summaries in `obs`, `var` or `uns` compute once at the end.
- Sparse arrays stay sparse, with missing values stored explicitly as NaN and implicit entries counting as zeros.
- Where an operation would move implicit zeros, for example centering, it raises a {class}`NotImplementedError` that says so instead of silently densifying the data.
- Results match the numpy result for every supported array type.
- Dask arrays with sparse chunks are not supported yet.

## What longitudinal data means for a function

- Per-variable statistics, for example in normalization, imputation and outlier handling, are computed across observations and timepoints, so a variable has the same scale at every timepoint.
- Elementwise operations, for example {func}`~ehrapy.preprocessing.log_norm` or {func}`~ehrapy.preprocessing.explicit_impute`, apply at every timepoint.
- Functions that need one value per observation and variable, such as PCA, embeddings, clustering, feature ranking, and most plots, only accept 2D data.
  Aggregate the time axis first with {func}`~ehrapy.preprocessing.summarize_measurements`, which turns a 3D array into a 2D array with one column per variable and statistic.

## Preprocessing support

| Function | numpy | sparse | dask | longitudinal data |
| --- | --- | --- | --- | --- |
| {func}`~ehrapy.preprocessing.scale_norm` | yes | with `with_mean=False` | lazy | statistics across observations and timepoints |
| {func}`~ehrapy.preprocessing.minmax_norm` | yes | no | lazy | statistics across observations and timepoints |
| {func}`~ehrapy.preprocessing.maxabs_norm` | yes | yes | lazy | statistics across observations and timepoints |
| {func}`~ehrapy.preprocessing.robust_scale_norm` | yes | with `with_centering=False` | lazy | statistics across observations and timepoints |
| {func}`~ehrapy.preprocessing.quantile_norm` | yes | no | lazy | quantiles across observations and timepoints |
| {func}`~ehrapy.preprocessing.power_norm` | yes | no | lazy | fit across observations and timepoints |
| {func}`~ehrapy.preprocessing.log_norm` | yes | with `offset=1` | lazy | elementwise |
| {func}`~ehrapy.preprocessing.offset_negative_values` | yes | without negative values | lazy | global minimum |
| {func}`~ehrapy.preprocessing.explicit_impute` | yes | yes | lazy | elementwise, or one value per timepoint |
| {func}`~ehrapy.preprocessing.simple_impute` | yes | yes | lazy | statistics across observations and timepoints |
| {func}`~ehrapy.preprocessing.knn_impute` | yes | no | no | timepoints are imputed as observations |
| {func}`~ehrapy.preprocessing.miss_forest_impute` | yes | no | no | timepoints are imputed as observations |
| {func}`~ehrapy.preprocessing.locf_impute` | yes | no | lazy | carries values forward in time, longitudinal data only |
| {func}`~ehrapy.preprocessing.missing_data_mask` | yes | yes | lazy | elementwise |
| {func}`~ehrapy.preprocessing.qc_metrics` | yes | yes | one compute | variable metrics across observations and timepoints, observation metrics across variables and timepoints |
| {func}`~ehrapy.preprocessing.qc_lab_measurements` | yes | yes | one compute | an observation is flagged if any timepoint is out of range |
| {func}`~ehrapy.preprocessing.mcar_test` | yes | no | no | 2D only |
| {func}`~ehrapy.preprocessing.filter_features`, {func}`~ehrapy.preprocessing.filter_observations` | yes | yes | one compute of the counts | counts across observations or variables and timepoints |
| {func}`~ehrapy.preprocessing.winsorize`, {func}`~ehrapy.preprocessing.clip_quantile` | yes | if the limits keep implicit zeros | lazy | limits across observations and timepoints |
| {func}`~ehrapy.preprocessing.variable_correlations` | yes | no | one compute | values aggregated over time first |
| {func}`~ehrapy.preprocessing.summarize_measurements` | yes | no | lazy for longitudinal data | aggregates the time axis into a 2D array |
| {func}`~ehrapy.preprocessing.detect_bias` | yes | no | no | 2D only |
| {func}`~ehrapy.preprocessing.encode` | yes | only when nothing needs encoding | one compute to find the categories | encodes every timepoint |
