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
- Functions that need feature types do not infer them from dask arrays, because inference reads every value; run {func}`ehrdata.infer_feature_types` first.
- Errors that depend on the values, for example a Box-Cox transform of non-positive data, appear when a lazy result is computed.

## What longitudinal data means for a function

- Per-variable statistics, for example in normalization, imputation and outlier handling, are computed across observations and timepoints, so a variable has the same scale at every timepoint.
- Elementwise operations, for example {func}`~ehrapy.preprocessing.log_norm` or {func}`~ehrapy.preprocessing.explicit_impute`, apply at every timepoint.
- Functions that need one value per observation and variable, such as PCA, t-SNE, feature ranking, and most plots, only accept 2D data.
  Aggregate the time axis first with {func}`~ehrapy.preprocessing.summarize_measurements`, which turns a 3D array into a 2D array with one column per variable and statistic.
- Functions that work on the neighbors graph, such as UMAP, Leiden clustering and PAGA, run on longitudinal data when the neighbors were computed with a time series distance, for example `ep.pp.neighbors(edata, metric="dtw", use_rep="tem_data")`.
  Embedding plots of such data can be colored by `obs` columns, but not by variables.
- Longitudinal plots and {func}`~ehrapy.tools.ncp` use the time axis and only accept 3D data.

## Preprocessing support

| Function | numpy | sparse | dask | longitudinal data |
| --- | --- | --- | --- | --- |
| {func}`~ehrapy.preprocessing.scale_norm` | yes | with `with_mean=False` | lazy | statistics across observations and timepoints |
| {func}`~ehrapy.preprocessing.minmax_norm` | yes | no | lazy | statistics across observations and timepoints |
| {func}`~ehrapy.preprocessing.maxabs_norm` | yes | yes | lazy | statistics across observations and timepoints |
| {func}`~ehrapy.preprocessing.robust_scale_norm` | yes | with `with_centering=False` | lazy | statistics across observations and timepoints |
| {func}`~ehrapy.preprocessing.quantile_norm` | yes | no | lazy | quantiles across observations and timepoints |
| {func}`~ehrapy.preprocessing.power_norm` | yes | no | lazy | fit across observations and timepoints |
| {func}`~ehrapy.preprocessing.log_norm` | yes | with `offset=1` | lazy, negative values become NaN instead of raising | elementwise |
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
| {func}`~ehrapy.preprocessing.combat` | yes | no | no | 2D only |
| {func}`~ehrapy.preprocessing.regress_out` | yes | no | no | 2D only |
| {func}`~ehrapy.preprocessing.sample` | yes | yes | lazy | samples observations |

## Tools and plots support

Tools, plots and getters that fit models or build tables densify, and compute, only the variables they use.
Functions that only read `obs`, `obsm`, `obsp` or `uns`, such as {func}`~ehrapy.tools.umap`, {func}`~ehrapy.tools.leiden`, {func}`~ehrapy.tools.paga`, {func}`~ehrapy.tools.stratified_table_one` or {func}`~ehrapy.plot.ncp`, work with every array type and with longitudinal data.

| Function | numpy | sparse | dask | longitudinal data |
| --- | --- | --- | --- | --- |
| {func}`~ehrapy.tools.rank_features_groups` | yes | yes | one compute | 2D only for variables, `obs` columns of longitudinal data can be ranked |
| {func}`~ehrapy.tools.filter_rank_features_groups` | yes | yes | one compute | 2D only |
| {func}`~ehrapy.tools.rank_features_supervised` | yes | yes | one compute | 2D only |
| {func}`~ehrapy.tools.ols`, {func}`~ehrapy.tools.glm`, {func}`~ehrapy.tools.kaplan_meier`, {func}`~ehrapy.tools.nelson_aalen`, {func}`~ehrapy.tools.weibull`, {func}`~ehrapy.tools.cox_ph`, {func}`~ehrapy.tools.weibull_aft`, {func}`~ehrapy.tools.log_logistic_aft`, {func}`~ehrapy.tools.cox_ph_adjusted_curves` | yes | yes | one compute | variables need 2D data, `obs` columns can be used with longitudinal data |
| {func}`~ehrapy.tools.iptw`, {func}`~ehrapy.tools.g_computation`, {func}`~ehrapy.tools.aipw`, {func}`~ehrapy.tools.propensity_score_matching`, {func}`~ehrapy.tools.t_learner`, {func}`~ehrapy.tools.s_learner`, {func}`~ehrapy.tools.x_learner`, {func}`~ehrapy.tools.covariate_balance`, {func}`~ehrapy.tools.positivity_check` | yes | yes | one compute | 2D only |
| {func}`~ehrapy.tools.tsne`, {func}`~ehrapy.tools.dendrogram` | yes | yes | one compute | 2D only, unless `use_rep` names an embedding |
| {func}`~ehrapy.tools.ingest` | yes | no | no | 2D only |
| {func}`~ehrapy.tools.famd` | yes | no | no | 2D only |
| {func}`~ehrapy.tools.ncp` | yes | no | one compute | longitudinal data only, decomposes observations, variables and time |
| {func}`~ehrapy.plot.timeseries` | yes | no | one compute of the plotted values | longitudinal data only, plots values over time |
| {func}`~ehrapy.plot.sankey_diagram_time` | yes | no | one compute of the plotted variable | longitudinal data only, shows transitions between consecutive timepoints |
| {func}`~ehrapy.plot.ncp_cluster_trajectories` | yes | no | one compute of the plotted means | longitudinal data only, plots mean trajectories per group |
| {func}`~ehrapy.plot.variable_correlations`, {func}`~ehrapy.plot.variable_dependencies` | yes | no | one compute | values aggregated over time first |
| {func}`~ehrapy.plot.missing_values_matrix`, {func}`~ehrapy.plot.missing_values_barplot`, {func}`~ehrapy.plot.missing_values_heatmap`, {func}`~ehrapy.plot.missing_values_dendrogram` | yes | yes | one compute of the missing value mask | 2D only |
| {func}`~ehrapy.plot.ols` | yes | yes | one compute | 2D only |
| {func}`~ehrapy.plot.heatmap`, {func}`~ehrapy.plot.dotplot`, {func}`~ehrapy.plot.matrixplot`, {func}`~ehrapy.plot.stacked_violin`, {func}`~ehrapy.plot.tracksplot`, {func}`~ehrapy.plot.violin`, {func}`~ehrapy.plot.clustermap`, {func}`~ehrapy.plot.scatter`, {func}`~ehrapy.plot.dendrogram` and the `rank_features_groups_*` plots | yes | yes | computes the plotted variables | 2D only |
| {func}`~ehrapy.plot.pca`, {func}`~ehrapy.plot.tsne`, {func}`~ehrapy.plot.umap`, {func}`~ehrapy.plot.diffmap`, {func}`~ehrapy.plot.draw_graph`, {func}`~ehrapy.plot.embedding`, {func}`~ehrapy.plot.paga`, {func}`~ehrapy.plot.paga_compare`, {func}`~ehrapy.plot.pca_overview` | yes | yes | computes the plotted variables | coloring by `obs` columns works for longitudinal data, coloring by variables needs 2D data |
| {func}`~ehrapy.plot.paga_path` | yes | yes | no | variables need 2D data |
| {func}`~ehrapy.plot.dpt_timeseries` | yes | no | one compute | 2D only |
| {func}`~ehrapy.get.obs_df` | yes | yes | one compute | variables need 2D data, `obs` columns can be read from longitudinal data |
| {func}`~ehrapy.get.var_df` | yes | yes | one compute | 2D only |

The plots built on scanpy read dask arrays through scanpy, which may compute the plotted variables more than once.
Plots that draw a dendrogram fail on scipy sparse matrices in scanpy; compute the dendrogram first with {func}`~ehrapy.tools.dendrogram` or use sparse arrays.
