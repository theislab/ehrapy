# Changelog

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## The future

### 🚀 Features

* {func}`ep.tl.comorbidity_index <ehrapy.tools.comorbidity_index>` scores every patient with the Charlson or Elixhauser comorbidity index from ICD-9-CM and ICD-10 codes, and from SNOMED codes through the ICD codes mapped to them, using the coding algorithms of Quan et al. 2005 with the Charlson, Quan 2011 or van Walraven weights, and stores the score and every comorbidity in `obs` ([#1161](https://github.com/theislab/ehrapy/pull/1161)) @Zethson
* {func}`ep.get.obs_df <ehrapy.get.obs_df>` and the functions that read variables through it, the survival and regression models, the causal estimators, {func}`ep.pl.scatter <ehrapy.plot.scatter>`, {func}`ep.pl.catplot <ehrapy.plot.catplot>` and {func}`ep.pl.ols <ehrapy.plot.ols>`, take variables of longitudinal 3D data at their first non-missing value, the baseline, or at another statistic over time passed to `obs_df` as `statistic` ([#1164](https://github.com/theislab/ehrapy/pull/1164)) @Zethson

  The survival models derive the duration and event from an `event_col` that is a longitudinal variable, so that `ep.tl.kaplan_meier(edata, event_col="SepsisLabel")` estimates the time to sepsis on PhysioNet 2019.
  The event is whether the variable is ever 1, at the time of its first 1 in `edata.tem["time_value"]`, and observations without one are censored at their last non-missing value.
  The fitted models hold the derived durations and events as `durations` and `event_observed`.
* {func}`ep.pp.summarize_measurements <ehrapy.preprocessing.summarize_measurements>` summarizes the timepoints selected by `tem_names`, or several named windows such as the first 6 and the last 24 hours in columns `{var}_{stat}_{window}`, and `ep.ml` computes the features of its observation windows with it ([#1159](https://github.com/theislab/ehrapy/pull/1159)) @Zethson
* {func}`ep.tl.kaplan_meier <ehrapy.tools.kaplan_meier>` estimates the cumulative incidence of an `event_of_interest` with the Aalen-Johansen estimator when the event column holds competing events, {func}`ep.tl.nelson_aalen <ehrapy.tools.nelson_aalen>` its cause-specific cumulative hazard, and {func}`ep.pl.kaplan_meier <ehrapy.plot.kaplan_meier>` plots the cumulative incidence ([#1155](https://github.com/theislab/ehrapy/pull/1155)) @Zethson

  Without `event_of_interest`, the univariate survival models raise an error for event columns with more than one event type instead of silently treating every event type as the same event.
* {func}`ep.pp.filter_observations <ehrapy.preprocessing.filter_observations>` defines a cohort by inclusion criteria in `query` on `obs` columns and variables, which for longitudinal data take the value at the timepoints selected by `tem_names` reduced by `agg`, and records every step in a {class}`ep.tl.CohortTracker <ehrapy.tools.CohortTracker>` passed as `tracker` for CONSORT-style flowcharts ([#1157](https://github.com/theislab/ehrapy/pull/1157)) @Zethson
* {func}`ep.pp.pca <ehrapy.preprocessing.pca>` and {func}`ep.tl.famd <ehrapy.tools.famd>` run on longitudinal data by unfolding every variable and timepoint into a feature, store loadings per variable and timepoint in `varm`, and {func}`ep.pp.neighbors <ehrapy.preprocessing.neighbors>` uses the resulting `X_pca` for longitudinal `.X` ([#1158](https://github.com/theislab/ehrapy/pull/1158)) @Zethson
* {func}`ep.pp.qc_metrics <ehrapy.preprocessing.qc_metrics>` adds, for longitudinal data, the share of observations that measure every variable and the median time between its consecutive values to `var`, and the number of measured timepoints with the first and last measured time to `obs`, in the times of `edata.tem[time_key]` ([#1156](https://github.com/theislab/ehrapy/pull/1156)) @Zethson
* {func}`ep.pp.qc_lab_measurements <ehrapy.preprocessing.qc_lab_measurements>` flags implausible jumps between consecutive values of longitudinal variables in `obs`, beyond an absolute or relative `max_change` per time for all or single variables, or else beyond the range of normal changes estimated with `method` ([#1156](https://github.com/theislab/ehrapy/pull/1156)) @Zethson
* Models of time series in `ep.ml` measure the time since the last observation with the times of the timepoints in `edata.tem[time_key]`, a new argument of {func}`ep.ml.fit <ehrapy.ml.fit>`, instead of counting timepoints, so that irregular timepoints are spaced correctly ([#1151](https://github.com/theislab/ehrapy/pull/1151)) @Zethson
* {class}`ep.ml.Task <ehrapy.ml.Task>` with `rolling=True` predicts a longitudinal variable, such as the hourly sepsis label of PhysioNet 2019, at every timepoint from the timepoints before it, and every function of `ep.ml` fits, stores, evaluates, calibrates and explains these predictions per timepoint while keeping patients together ([#1152](https://github.com/theislab/ehrapy/pull/1152)) @Zethson

  On the held-out patients of PhysioNet 2019, a GRU predicts the hourly sepsis label from the hours before with an AUROC of 0.75 and AUPRC of 0.05, and gradient boosting on summaries of those hours with 0.67 and 0.02, for 1.2% positive hours.
* {func}`ep.pp.summarize_measurements <ehrapy.preprocessing.summarize_measurements>` computes the number of values, their standard deviation and, for longitudinal data, their slope over time ([#1150](https://github.com/theislab/ehrapy/pull/1150)) @Zethson
* {func}`ep.pl.trajectories <ehrapy.plot.trajectories>` plots the mean of longitudinal variables over time with a confidence band for every group of observations ([#1146](https://github.com/theislab/ehrapy/pull/1146)) @Zethson
* {func}`ep.pp.locf_impute <ehrapy.preprocessing.locf_impute>` takes a `limit` on how many timepoints an observed value is carried forward, and its fallback only fills timepoints before a patient's first observation ([#1147](https://github.com/theislab/ehrapy/pull/1147)) @Zethson
* `ep.ml` fits, applies and evaluates models that predict patient outcomes from static and longitudinal data ([#1144](https://github.com/theislab/ehrapy/pull/1144)) @Zethson

  {func}`ep.ml.split <ehrapy.ml.split>` assigns patients to `train`, `tuning` and `held_out` sets, at random or by time, and {func}`ep.ml.fit <ehrapy.ml.fit>` fits imputation, scaling and the model on `train` only.
  A {class}`ep.ml.Task <ehrapy.ml.Task>` sets binary, multiclass, multilabel, regression or survival targets and the observation window that longitudinal variables are summarized over.
  {func}`ep.ml.fit <ehrapy.ml.fit>` trains linear, gradient boosting, random forest and Cox models, multilayer perceptrons, and GRU, LSTM, GRU-D, TCN, transformer and RETAIN models of time series, whose dependencies the new `ml` extra installs.
  {func}`ep.ml.evaluate <ehrapy.ml.evaluate>` reports the metrics of every kind of task with confidence intervals from resampled patients, overall, per subgroup and as differences between subgroups such as demographic parity and equalized odds.
  {func}`ep.ml.calibrate <ehrapy.ml.calibrate>` calibrates predicted probabilities with Platt scaling, isotonic regression or temperature scaling, and {func}`ep.ml.conformalize <ehrapy.ml.conformalize>` adds conformal prediction sets and intervals, both on the tuning set.
  {func}`ep.ml.permutation_importance <ehrapy.ml.permutation_importance>` stores how much every variable matters to a model in `var` and `varm`.
  {func}`ep.pl.prediction_performance <ehrapy.plot.prediction_performance>` plots the ROC, precision-recall and calibration curves, per class or label for multiclass and multilabel tasks, and {func}`ep.pl.subgroup_performance <ehrapy.plot.subgroup_performance>` the metrics of every subgroup.
  On PhysioNet 2012, a GRU on the first 48 hours predicts in-hospital mortality with a held-out AUROC of 0.87 and AUPRC of 0.57, and gradient boosting on their summaries with 0.86 and 0.51.
* {func}`ep.pp.gradient_boosting_impute <ehrapy.preprocessing.gradient_boosting_impute>` imputes every variable with a gradient boosting model, which for longitudinal data also uses the closest observed values before and after each timepoint @Zethson

  With every variable held out at 10% of the observed hours of all 11,988 PhysioNet 2012 patients, its RMSE on the standardized held-out values is 0.62 in 36 seconds, against 1.09 in 232 seconds for {func}`ep.pp.miss_forest_impute <ehrapy.preprocessing.miss_forest_impute>` and 0.82 for {func}`ep.pp.locf_impute <ehrapy.preprocessing.locf_impute>`.
* Preprocessing functions support numpy, scipy sparse and dask arrays, including dask arrays with sparse chunks, for static 2D and longitudinal 3D data ([#1132](https://github.com/theislab/ehrapy/pull/1132)) @Zethson

  Sparse arrays stay sparse and dask arrays stay lazy, and functions that store summaries compute once.
  The few combinations that cannot work this way raise a `NotImplementedError` that says why, such as centering or ComBat on sparse data, the faiss backend of {func}`ep.pp.knn_impute <ehrapy.preprocessing.knn_impute>` on sparse or dask arrays, and {func}`ep.pp.miss_forest_impute <ehrapy.preprocessing.miss_forest_impute>` or {func}`ep.pp.detect_bias <ehrapy.preprocessing.detect_bias>` on dask arrays.
* {func}`ep.pp.summarize_measurements <ehrapy.preprocessing.summarize_measurements>` aggregates longitudinal data over time into a 2D object with one column per variable and statistic (`min`, `max`, `mean`, `median`, `first`, `last`), which makes every 2D-only function usable on longitudinal data ([#1132](https://github.com/theislab/ehrapy/pull/1132)) @Zethson
* {func}`ep.pp.winsorize <ehrapy.preprocessing.winsorize>`, {func}`ep.pp.clip_quantile <ehrapy.preprocessing.clip_quantile>` and {func}`ep.pp.qc_lab_measurements <ehrapy.preprocessing.qc_lab_measurements>` support longitudinal data ([#1132](https://github.com/theislab/ehrapy/pull/1132)) @Zethson
* Tools, plots and `ep.get` functions support numpy, scipy sparse and dask arrays, and densify and compute only the variables they use ([#1134](https://github.com/theislab/ehrapy/pull/1134)) @Zethson

  The causal estimators, {func}`ep.tl.famd <ehrapy.tools.famd>` and {func}`ep.tl.rank_features_supervised <ehrapy.tools.rank_features_supervised>` accept sparse and dask arrays instead of rejecting or fully densifying them.
  Embedding plots such as {func}`ep.pl.umap <ehrapy.plot.umap>` color longitudinal data by `obs` columns, and {func}`ep.pl.timeseries <ehrapy.plot.timeseries>` plots 3D `.X` with `layer=None`.

### 💥 Breaking changes

* Remove the single-cell leftovers `feature_symbols`, `use_raw` and the unused fields of `ep.settings` ([#1154](https://github.com/theislab/ehrapy/pull/1154), [#1162](https://github.com/theislab/ehrapy/pull/1162)) @Zethson

  The `feature_symbols` argument of the `ep.pl` and `ep.get` functions and the `use_raw` argument of the `ep.pl` functions are gone, and plots always read `.X` or `layer`, never `.raw`.
  `ep.settings` keeps `verbosity` and `n_jobs`, the only fields ehrapy reads, and drops `plot_suffix`, `file_format_data`, `file_format_figs`, `autosave`, `autoshow`, `writedir`, `cachedir`, `datasetdir`, `figdir`, `cache_compression`, `max_memory`, `categories_to_ignore` and `n_pcs`.
* The `ep.pl.missing_values_*` plots are interactive HoloViews plots that accept longitudinal data, and the `missingno` dependency is dropped ([#1149](https://github.com/theislab/ehrapy/pull/1149)) @Zethson

  {func}`ep.pl.missing_values_matrix <ehrapy.plot.missing_values_matrix>` shows the percentage of observed values per observation, or per timepoint for longitudinal data, and the bar plot, heatmap and dendrogram count every timepoint of every observation as one row.
  They take `var_names`, `width`, `height` and `title` instead of the `missingno` arguments.
* {func}`ep.pl.timeseries <ehrapy.plot.timeseries>` plots the 3D `.X` by default instead of `.layers["tem_data"]`, like every other function ([#1121](https://github.com/theislab/ehrapy/pull/1121)) @sueoglu
* Survival analysis and regression models take their columns from `edata.obs` or variables and only drop observations missing a column the model uses ([#1133](https://github.com/theislab/ehrapy/pull/1133)) @Zethson

  {func}`ep.tl.kaplan_meier <ehrapy.tools.kaplan_meier>`, {func}`ep.tl.cox_ph <ehrapy.tools.cox_ph>` and the other survival fitters, {func}`ep.tl.ols <ehrapy.tools.ols>` and {func}`ep.tl.glm <ehrapy.tools.glm>` accept obs columns wherever they accept variables, and work on longitudinal data when every column they use lives in `obs`.
  Previously, every observation with a missing value in any variable was silently dropped, even in variables the model did not use.
  Regression fitters gained `covariates`, the univariate fitters take `entry_col` and `weights_col` instead of the `entry` and `weights` arrays, and numeric columns are passed to the models as numbers, so a binomial {func}`ep.tl.glm <ehrapy.tools.glm>` on a 0/1 outcome models the probability of 1.
* Normalization functions take explicit parameters instead of forwarding `**kwargs` to scikit-learn or dask-ml ([#1132](https://github.com/theislab/ehrapy/pull/1132)) @Zethson

  {func}`ep.pp.scale_norm <ehrapy.preprocessing.scale_norm>` takes `with_mean` and `with_std`, {func}`ep.pp.minmax_norm <ehrapy.preprocessing.minmax_norm>` `feature_range`, {func}`ep.pp.robust_scale_norm <ehrapy.preprocessing.robust_scale_norm>` `with_centering`, `with_scaling`, `quantile_range` and `unit_variance`, {func}`ep.pp.quantile_norm <ehrapy.preprocessing.quantile_norm>` `n_quantiles`, `output_distribution`, `subsample` and `random_state`, and {func}`ep.pp.power_norm <ehrapy.preprocessing.power_norm>` `method` and `standardize`.
  A `groupby` column with missing values now raises instead of leaving those observations unnormalized, and the `dask` extra no longer installs dask-ml.
* {func}`ep.pp.winsorize <ehrapy.preprocessing.winsorize>` takes `inclusive` instead of `**kwargs`, ignores missing values when ranking, and cuts 1% from each side by default ([#1132](https://github.com/theislab/ehrapy/pull/1132)) @Zethson

  The previous default `limits=(0.01, 0.99)` cut 99% of the largest values, replacing almost every value of a variable with the same number.
* {func}`ep.pp.regress_out <ehrapy.preprocessing.regress_out>` no longer takes `n_jobs` ([#1132](https://github.com/theislab/ehrapy/pull/1132)) @Zethson
* {func}`ep.pp.summarize_measurements <ehrapy.preprocessing.summarize_measurements>` no longer accepts `statistics=None`, which always raised ([#1132](https://github.com/theislab/ehrapy/pull/1132)) @Zethson
* Remove `ep.pp.mice_forest_impute` and drop the `miceforest` dependency @Zethson

  `miceforest` is effectively unmaintained (last commit 2025-10-27) and broken against `lightgbm>=4.7.0`, which it calls through a private, name-mangled internal ([miceforest#104](https://github.com/AnotherSamWilson/miceforest/issues/104)).
  Use {func}`ep.pp.gradient_boosting_impute <ehrapy.preprocessing.gradient_boosting_impute>` or {func}`ep.pp.miss_forest_impute <ehrapy.preprocessing.miss_forest_impute>` instead.
  For a LightGBM backend, pass `IterativeImputer(estimator=LGBMRegressor(...))` directly.
* Unify the API conventions across `ep.pp`, `ep.tl`, `ep.pl` and `ep.get` ([#1126](https://github.com/theislab/ehrapy/pull/1126)) @Zethson

  Required arguments are positional and every argument with a default is keyword-only.
  Grouping keys are called `groupby` (was `group_key`, `cluster_key`, `balanced_key`), feature subsets `var_names` (was `vars`, `input_features`, `feature_names`), feature-name columns `feature_symbols` (was `gene_symbols`, `features`), and result keys `key_added` when written and `key` when read (was `uns_key`).
  {func}`ep.pp.combat <ehrapy.preprocessing.combat>` takes `batch_key` (was `key`), and {func}`ep.pp.pca <ehrapy.preprocessing.pca>` and {func}`ep.pp.sample <ehrapy.preprocessing.sample>` take `edata` (was `data`).
* Replace `inplace` with `copy` ([#1126](https://github.com/theislab/ehrapy/pull/1126)) @Zethson

  {func}`ep.pp.combat <ehrapy.preprocessing.combat>`, {func}`ep.pp.highly_variable_features <ehrapy.preprocessing.highly_variable_features>`, {func}`ep.tl.dendrogram <ehrapy.tools.dendrogram>` and {func}`ep.tl.ingest <ehrapy.tools.ingest>` take `copy` instead of `inplace`.
  {func}`ep.pp.qc_metrics <ehrapy.preprocessing.qc_metrics>`, {func}`ep.pp.detect_bias <ehrapy.preprocessing.detect_bias>`, {func}`ep.tl.embedding_density <ehrapy.tools.embedding_density>`, {func}`ep.tl.filter_rank_features_groups <ehrapy.tools.filter_rank_features_groups>`, {func}`ep.tl.rank_features_supervised <ehrapy.tools.rank_features_supervised>` and {func}`ep.tl.cox_ph_adjusted_curves <ehrapy.tools.cox_ph_adjusted_curves>` gained `copy`.
  `qc_metrics` and `detect_bias` store their results in `edata` instead of returning them (`detect_bias` under `uns["bias"]`), and `rank_features_supervised` stores the model's test score in `uns[key_added]` instead of returning it.
* Remove the deprecated `ep.tl.kmf`, `ep.pp.subsample` and the `n_neighbours` alias of {func}`ep.pp.knn_impute <ehrapy.preprocessing.knn_impute>` ([#1126](https://github.com/theislab/ehrapy/pull/1126)) @Zethson

### 🐛 Bug Fixes

* {func}`ep.tl.cox_ph_adjusted_curves <ehrapy.tools.cox_ph_adjusted_curves>` raises an error when `strata` is neither a covariate nor one of the `strata` of `cph`, which gave every group the same curve, and its documentation no longer asks to leave `strata` out of the model ([#1192](https://github.com/theislab/ehrapy/pull/1192)) @Zethson
* {func}`ep.pp.qc_metrics <ehrapy.preprocessing.qc_metrics>` and {func}`ep.pp.qc_lab_measurements <ehrapy.preprocessing.qc_lab_measurements>` measure the longitudinal metrics and changes per time of 3D data in the unit of `edata.tem["time_value"]`, such as days on `ed.dt.mimic_iv_meds`, instead of in seconds of `edata.tem["interval_start_offset"]`, which they keep using when `tem` has no `time_value` ([#1202](https://github.com/theislab/ehrapy/pull/1202)) @Zethson
* {func}`ep.pl.cox_ph_forestplot <ehrapy.plot.cox_ph_forestplot>` draws the confidence intervals as horizontal error bars, puts the reference line at a coefficient of 0, which is a hazard ratio of 1, instead of 1, and gives the header its own row instead of overlapping the top coefficient ([#1191](https://github.com/theislab/ehrapy/pull/1191)) @Zethson
* The survival models that derive durations from a longitudinal `event_col` measure them in the unit of `edata.tem["time_value"]`, such as hours on PhysioNet 2012 and 2019, instead of in seconds of `edata.tem["interval_start_offset"]`, which they keep using when `tem` has no `time_value` ([#1190](https://github.com/theislab/ehrapy/pull/1190)) @Zethson
* The examples of {func}`ep.tl.kaplan_meier <ehrapy.tools.kaplan_meier>`, {func}`ep.tl.nelson_aalen <ehrapy.tools.nelson_aalen>`, {func}`ep.tl.weibull <ehrapy.tools.weibull>`, {func}`ep.tl.weibull_aft <ehrapy.tools.weibull_aft>`, {func}`ep.tl.log_logistic_aft <ehrapy.tools.log_logistic_aft>`, {func}`ep.tl.cox_ph <ehrapy.tools.cox_ph>` and {func}`ep.pl.kaplan_meier <ehrapy.plot.kaplan_meier>` no longer flip `censor_flg` of MIMIC-II, which `ed.dt.mimic_2` already codes as 1 for death, so that they model the time to death instead of the time to censoring ([#1189](https://github.com/theislab/ehrapy/pull/1189)) @Zethson
* The logistic propensity model of the causal estimators and diagnostics standardizes the covariates before fitting, so that estimates no longer depend on the units of the covariates and the fit converges on unscaled data such as the 21 baseline covariates of MIMIC-II ([#1186](https://github.com/theislab/ehrapy/pull/1186)) @Zethson
* {func}`ep.pp.neighbors <ehrapy.preprocessing.neighbors>` with `metric="dtw"`, `"soft_dtw"` or `"gak"` stores the name of the metric in `uns["neighbors"]["params"]` instead of the internal distance function together with the whole time series array, so that {func}`ed.io.write_h5ed <ehrdata.io.write_h5ed>` writes the result ([#1188](https://github.com/theislab/ehrapy/pull/1188)) @Zethson
* {func}`ep.pl.violin <ehrapy.plot.violin>` no longer draws a legend that repeats the x axis when `groupby` has integer categories such as a 0/1 treatment flag ([#1182](https://github.com/theislab/ehrapy/pull/1182)) @Zethson
* {meth}`CausalEstimate.summary <ehrapy.tools.CausalEstimate.summary>` labels the effect with its new `estimand` attribute, so that {func}`ep.tl.propensity_score_matching <ehrapy.tools.propensity_score_matching>` with `target="att"` reports an ATT instead of an ATE (PRLINK) @Zethson
* The `summary()` of a {class}`ep.tl.CausalEstimate <ehrapy.tools.CausalEstimate>` labels the effect with its new `estimand` attribute, so that {func}`ep.tl.propensity_score_matching <ehrapy.tools.propensity_score_matching>` with `target="att"` reports an ATT instead of an ATE ([#1183](https://github.com/theislab/ehrapy/pull/1183)) @Zethson
* {func}`ep.tl.s_learner <ehrapy.tools.s_learner>`, {func}`ep.tl.t_learner <ehrapy.tools.t_learner>` and {func}`ep.tl.x_learner <ehrapy.tools.x_learner>` take a `random_state` that seeds their built-in gradient boosting and random forest models, and the effect estimators pass their `random_state` to these models too, so that their estimates are reproducible ([#1184](https://github.com/theislab/ehrapy/pull/1184)) @Zethson
* The causal estimators and diagnostics such as {func}`ep.tl.covariate_balance <ehrapy.tools.covariate_balance>` treat numeric variables of an object `X`, such as that of `ed.dt.mimic_2()`, as numeric instead of one-hot encoding each of their values ([#1185](https://github.com/theislab/ehrapy/pull/1185)) @Zethson
* {func}`ep.pp.explicit_impute <ehrapy.preprocessing.explicit_impute>` no longer warns for every variable missing from a `replacement` mapping, because leaving variables out of it is the intended way to impute only some of them ([#1178](https://github.com/theislab/ehrapy/pull/1178)) @Zethson
* {func}`ep.tl.rank_features_groups <ehrapy.tools.rank_features_groups>` reports the log2 ratio of the fractions of observations with value 1 as log fold change of binary categorical features and NaN for categorical features with more levels, instead of 1 regardless of direction, and orders features with equal adjusted p-values by their p-values ([#1180](https://github.com/theislab/ehrapy/pull/1180)) @Zethson
* {func}`ep.tl.rank_features_groups <ehrapy.tools.rank_features_groups>` tests numeric variables with missing values on their observed values instead of giving them a score of 0 and a p-value of 1 with t-tests, or ranking missing values as the largest with the Wilcoxon test, and computes their log fold changes from the observed values instead of returning NaN ([#1179](https://github.com/theislab/ehrapy/pull/1179)) @Zethson
* {func}`ep.ml.calibrate <ehrapy.ml.calibrate>` defaults to `method="platt"` instead of `"isotonic"`, because isotonic regression overfits tuning sets of typical size, such as on PhysioNet 2012 with a calibration slope of 0.54 against 0.95 with Platt scaling ([#1169](https://github.com/theislab/ehrapy/pull/1169)) @Zethson
* {func}`ep.ml.evaluate <ehrapy.ml.evaluate>` and {func}`ep.pl.subgroup_performance <ehrapy.plot.subgroup_performance>` keep the subgroups of `groupby` in the order of its categories, or of their appearance, instead of sorting them ([#1169](https://github.com/theislab/ehrapy/pull/1169)) @Zethson
* {func}`ep.pp.summarize_measurements <ehrapy.preprocessing.summarize_measurements>` returns NaN without RuntimeWarnings for variables without values in a window of longitudinal data ([#1169](https://github.com/theislab/ehrapy/pull/1169)) @Zethson
* {func}`ep.pp.gradient_boosting_impute <ehrapy.preprocessing.gradient_boosting_impute>` leaves out predictors observed too rarely to split on instead of crashing when their few values all fall into the validation split of a model trained on more than 10,000 rows, and warns for variables that stay missing because they have no observed values in the training observations ([#1174](https://github.com/theislab/ehrapy/pull/1174)) @Zethson
* {func}`ep.pp.winsorize <ehrapy.preprocessing.winsorize>` and {func}`ep.pp.clip_quantile <ehrapy.preprocessing.clip_quantile>` without `var_names` and `obs_cols` apply to all numeric variables instead of silently changing nothing ([#1173](https://github.com/theislab/ehrapy/pull/1173)) @Zethson
* {func}`ep.pp.mcar_test <ehrapy.preprocessing.mcar_test>` tests longitudinal data at the timepoint selected by `tem_names` or reduced by `agg` over several timepoints, and raises an error naming the variables that are not observed together in at least two observations instead of returning a `NaN` p-value for Little's test ([#1176](https://github.com/theislab/ehrapy/pull/1176)) @Zethson
* {meth}`ep.tl.CohortTracker.plot_cohort_barplot <ehrapy.tools.CohortTracker.plot_cohort_barplot>` with `show=True` lays out its rows next to the legend instead of squeezing them together by the height of the legend ([#1177](https://github.com/theislab/ehrapy/pull/1177)) @Zethson
* Matplotlib figures in notebooks display inline after a holoviews-based ehrapy plot that runs before the first matplotlib figure, without `%matplotlib inline` ([#1168](https://github.com/theislab/ehrapy/pull/1168)) @Zethson
* {func}`ep.tl.ols <ehrapy.tools.ols>` and {func}`ep.tl.glm <ehrapy.tools.glm>` find the `obs` columns and variables quoted as `Q('...')` in a formula, so that names with spaces work again, by requiring formulaic 1.2.2 ([#1165](https://github.com/theislab/ehrapy/pull/1165)) @Zethson
* {func}`ep.pp.pca <ehrapy.preprocessing.pca>` uses scanpy's dask-native `covariance_eigh` solver for dask arrays by default instead of requiring dask-ml ([#1163](https://github.com/theislab/ehrapy/pull/1163)) @Zethson
* The `rapids12` and `rapids13` extras install the CUDA wheels of rapids-singlecell, whose 0.18 release only builds from source under its old name and extras ([#1153](https://github.com/theislab/ehrapy/pull/1153)) @Zethson
* In {func}`ep.tl.rank_features_groups <ehrapy.tools.rank_features_groups>`, the `logfoldchanges` of numeric features are the log2 ratio of the group mean to the reference mean instead of scanpy's fold change for log1p-transformed data, and categorical features with `reference="rest"` and a `groups` subset are compared to all other observations like numeric features ([#1148](https://github.com/theislab/ehrapy/pull/1148)) @Zethson

  The previous fold changes were meaningless on EHR data and `NaN` with a `RuntimeWarning` on standardized data, and observations outside `groups` were counted with the tested group.
* {func}`ep.tl.rank_features_groups <ehrapy.tools.rank_features_groups>` stores results only for the compared groups like scanpy, instead of also for the `reference` and for groups outside `groups`, which drew empty panels in {func}`ep.pl.rank_features_groups <ehrapy.plot.rank_features_groups>`, and no longer crashes on categorical features with a `groups` subset ([#1145](https://github.com/theislab/ehrapy/pull/1145)) @Zethson
* {func}`ep.pl.violin <ehrapy.plot.violin>`, {func}`ep.pl.scatter <ehrapy.plot.scatter>`, {func}`ep.pl.ols <ehrapy.plot.ols>` and the causal estimators and diagnostics such as {func}`ep.tl.iptw <ehrapy.tools.iptw>` work on 3D data when they only use `obs` columns, and {func}`ep.pl.dendrogram <ehrapy.plot.dendrogram>` when the dendrogram is precomputed ([#1143](https://github.com/theislab/ehrapy/pull/1143)) @Zethson
* {func}`ep.pp.neighbors <ehrapy.preprocessing.neighbors>` uses the 3D `.X` by default with the time series metrics and rejects it with a clear error for other metrics instead of failing inside scikit-learn, and {func}`ep.tl.rank_features_groups <ehrapy.tools.rank_features_groups>` rejects 3D data with the same error as other 2D-only functions ([#1143](https://github.com/theislab/ehrapy/pull/1143)) @Zethson
* {func}`ep.pp.simple_impute <ehrapy.preprocessing.simple_impute>`, {func}`ep.pp.miss_forest_impute <ehrapy.preprocessing.miss_forest_impute>` and {func}`ep.pp.locf_impute <ehrapy.preprocessing.locf_impute>` raise a `KeyError` for unknown `var_names` instead of imputing the last variable, and {func}`ep.pp.explicit_impute <ehrapy.preprocessing.explicit_impute>` raises one for unknown `replacement` keys instead of ignoring them ([#1140](https://github.com/theislab/ehrapy/pull/1140)) @Zethson

  {func}`ep.pp.knn_impute <ehrapy.preprocessing.knn_impute>` and the normalization functions raise a `KeyError` for unknown `var_names` instead of reporting them as non-numeric.
* Matplotlib figures in notebooks no longer disappear after the first holoviews-based ehrapy plot @Zethson
* {func}`ep.pp.knn_impute <ehrapy.preprocessing.knn_impute>` imputes all numeric variables by default instead of raising on data with encoded categorical variables ([#1139](https://github.com/theislab/ehrapy/pull/1139)) @Zethson
* {func}`ep.pp.miss_forest_impute <ehrapy.preprocessing.miss_forest_impute>` keeps only one forest in memory instead of every forest it fitted, cutting peak memory about sixfold, and is reproducible for a fixed `random_state` @Zethson
* On dask arrays with missing values, {func}`ep.pp.minmax_norm <ehrapy.preprocessing.minmax_norm>` and {func}`ep.pp.robust_scale_norm <ehrapy.preprocessing.robust_scale_norm>` returned all-NaN variables and {func}`ep.pp.quantile_norm <ehrapy.preprocessing.quantile_norm>` returned wrong values ([#1132](https://github.com/theislab/ehrapy/pull/1132)) @Zethson
* {func}`ep.pp.filter_features <ehrapy.preprocessing.filter_features>` and {func}`ep.pp.filter_observations <ehrapy.preprocessing.filter_observations>` crashed for every non-numpy array, {func}`ep.pp.encode <ehrapy.preprocessing.encode>` crashed on sparse arrays, and the imputers crashed on variables without any observed value ([#1132](https://github.com/theislab/ehrapy/pull/1132)) @Zethson
* {func}`ep.pp.qc_metrics <ehrapy.preprocessing.qc_metrics>` no longer computes dask arrays once per variable or reports all-NaN statistics when any variable holds strings ([#1132](https://github.com/theislab/ehrapy/pull/1132)) @Zethson
* {func}`ep.tl.rank_features_groups <ehrapy.tools.rank_features_groups>` no longer computes dask arrays once per group statistic or crashes with `pts=True` on sparse arrays, on 3D data with a single timepoint, or with `num_cols_method="logreg"`, and its `pts` only hold the compared groups ([#1134](https://github.com/theislab/ehrapy/pull/1134)) @Zethson
* {func}`ep.tl.filter_rank_features_groups <ehrapy.tools.filter_rank_features_groups>` no longer crashes on sparse arrays and {func}`ep.tl.dendrogram <ehrapy.tools.dendrogram>` no longer crashes on scipy sparse matrices ([#1134](https://github.com/theislab/ehrapy/pull/1134)) @Zethson
* {func}`ep.tl.tsne <ehrapy.tools.tsne>`, {func}`ep.tl.dendrogram <ehrapy.tools.dendrogram>`, {func}`ep.tl.ingest <ehrapy.tools.ingest>`, the `ep.pl.rank_features_groups_*` plots, {func}`ep.pl.dpt_timeseries <ehrapy.plot.dpt_timeseries>` and embedding plots colored by variables reject 3D data with a clear error instead of failing inside scikit-learn or scanpy ([#1134](https://github.com/theislab/ehrapy/pull/1134)) @Zethson
* The `ep.pl.missing_values_*` plots read only the missing value mask of the plotted variables instead of densifying or computing the whole data matrix ([#1134](https://github.com/theislab/ehrapy/pull/1134)) @Zethson
* {func}`ep.pl.timeseries <ehrapy.plot.timeseries>`, {func}`ep.pl.sankey_diagram_time <ehrapy.plot.sankey_diagram_time>` and {func}`ep.pl.ncp_cluster_trajectories <ehrapy.plot.ncp_cluster_trajectories>` compute only the plotted values, `sankey_diagram_time` no longer crashes on dask arrays or on a single timepoint, and {func}`ep.tl.ncp <ehrapy.tools.ncp>` reports 2D sparse input as not 3D ([#1134](https://github.com/theislab/ehrapy/pull/1134)) @Zethson
* `ep.pp.explicit_impute()` now accepts falsy mapping replacement values such as `0`, `0.0`, and empty strings ([#1087](https://github.com/theislab/ehrapy/pull/1087)) @driavysinus
* `ep.pp.knn_impute()` now raises a clear `NotImplementedError` for unsupported array types (dask and sparse arrays) instead of failing silently ([#1109](https://github.com/theislab/ehrapy/pull/1109)) @sueoglu
* `_little_mcar_test` now computes its global covariance matrix with true pairwise deletion instead of centering on the global mean, fixing incorrect p-values under moderate-to-high missingness ([#1110](https://github.com/theislab/ehrapy/pull/1110)) @sueoglu
* {func}`ep.pp.encode <ehrapy.preprocessing.encode>` and {func}`ep.pp.clip_quantile(copy=True) <ehrapy.preprocessing.clip_quantile>` no longer modify their input ([#1126](https://github.com/theislab/ehrapy/pull/1126)) @Zethson
* {func}`ep.tl.filter_rank_features_groups <ehrapy.tools.filter_rank_features_groups>` no longer raises `KeyError: 'use_raw'` ([#1126](https://github.com/theislab/ehrapy/pull/1126)) @Zethson
* {func}`ep.tl.rank_features_supervised <ehrapy.tools.rank_features_supervised>` reports R² instead of accuracy for numeric targets ([#1126](https://github.com/theislab/ehrapy/pull/1126)) @Zethson
* `ep.tl` no longer leaks implementation details such as `np` and `sc`, and its `__all__` now lists `leiden`, `dendrogram`, `dpt`, `paga` and `ingest`; `ep.pl` gained an `__all__` ([#1126](https://github.com/theislab/ehrapy/pull/1126)) @Zethson
* {func}`ep.tl.famd <ehrapy.tools.famd>` works on 2D `.X` and layers with numeric or mixed variables, rejects 3D data, and stores per-variable loadings in `.varm` and all category loadings in `.uns` ([#1129](https://github.com/theislab/ehrapy/pull/1129)) @Zethson
* {func}`ep.tl.ncp <ehrapy.tools.ncp>` raises a `ValueError` for missing or negative values instead of returning NaN or negative factors ([#1129](https://github.com/theislab/ehrapy/pull/1129)) @Zethson
* {func}`ep.pl.ols <ehrapy.plot.ols>` gained `layer`, supports sparse arrays and rejects 3D data instead of flattening the time axis into extra points ([#1129](https://github.com/theislab/ehrapy/pull/1129)) @Zethson
* {func}`ep.get.obs_df <ehrapy.get.obs_df>` raises a clear error when reading variables from 3D data and {func}`ep.get.var_df <ehrapy.get.var_df>` for any 3D data, instead of failing inside pandas ([#1129](https://github.com/theislab/ehrapy/pull/1129)) @Zethson
* {func}`ep.tl.stratified_table_one <ehrapy.tools.stratified_table_one>` stores its table with `variable` and `level` columns so that results can be written to h5ad ([#1129](https://github.com/theislab/ehrapy/pull/1129)) @Zethson
* {func}`ep.tl.rank_features_groups <ehrapy.tools.rank_features_groups>` supports categorical features in sparse arrays ([#1129](https://github.com/theislab/ehrapy/pull/1129)) @Zethson
* {func}`ep.pp.neighbors <ehrapy.preprocessing.neighbors>` with a time series metric no longer makes patients without comparable measurements everyone's nearest neighbours but leaves them unconnected ([#1129](https://github.com/theislab/ehrapy/pull/1129)) @Zethson
* Align all scanpy wrappers with scanpy 1.12: the `ep.pl.rank_features_groups_*` plots read the {func}`ep.tl.rank_features_groups <ehrapy.tools.rank_features_groups>` results by default, which now honours `n_features` and stores `pts` per feature, {func}`ep.pp.pca <ehrapy.preprocessing.pca>` uses `var["highly_variable"]` by default as documented, every plot passes `feature_symbols` on to scanpy, and the deprecated `save`, `ep.tl.umap(method=...)` and the no-op `n_bins` of {func}`ep.pp.highly_variable_features <ehrapy.preprocessing.highly_variable_features>` are removed ([#1130](https://github.com/theislab/ehrapy/pull/1130)) @Zethson

### 📖 Documentation

* Add general introductions to the API sections and redirect removed tutorials to the tutorial gallery ([#1170](https://github.com/theislab/ehrapy/pull/1170)) @Zethson
* Refresh the README and installation guide (optional extras, install from GitHub) and drop dead Sphinx extensions ([#1128](https://github.com/theislab/ehrapy/pull/1128)) @Zethson
* Add imputation methods tutorial notebook, benchmarking six imputation strategies on the PhysioNet2012 dataset ([#1101](https://github.com/theislab/ehrapy/pull/1101)) @sueoglu
* Document the `ep.get` module, {func}`ep.tl.famd <ehrapy.tools.famd>` and {func}`ep.tl.anova_glm <ehrapy.tools.anova_glm>` ([#1126](https://github.com/theislab/ehrapy/pull/1126)) @Zethson
* Fix the {func}`ep.pl.kaplan_meier <ehrapy.plot.kaplan_meier>` example ([#1129](https://github.com/theislab/ehrapy/pull/1129)) @Zethson
* Describe {func}`ep.tl.ncp <ehrapy.tools.ncp>` in plain words and drop the doubled period in the docs footer ([#1131](https://github.com/theislab/ehrapy/pull/1131)) @Zethson
* Docstring examples keep time series in the 3D `.X` instead of `.layers["tem_data"]` and show the outputs the examples actually produce ([#1098](https://github.com/theislab/ehrapy/pull/1098)) @sueoglu
* Describe ehrapy's data sources and analyses in the README, docs landing page and package metadata, and add a longitudinal quickstart, a sitemap, an `llms.txt`, stable canonical URLs and a `CITATION.cff` ([#1141](https://github.com/theislab/ehrapy/pull/1141)) @Zethson
* Fix the docstring examples that no longer ran and add examples to {func}`ep.pp.pca <ehrapy.preprocessing.pca>`, {func}`ep.pp.neighbors <ehrapy.preprocessing.neighbors>`, {func}`ep.tl.umap <ehrapy.tools.umap>`, {func}`ep.tl.tsne <ehrapy.tools.tsne>`, {func}`ep.tl.leiden <ehrapy.tools.leiden>` and {func}`ep.tl.paga <ehrapy.tools.paga>` ([#1142](https://github.com/theislab/ehrapy/pull/1142)) @Zethson

  The tests now run every docstring example with xdoctest, so examples can no longer break silently.

### 🧰 Maintenance

* Run the imputation, causal inference and effect estimation tutorials in the notebook CI ([#1120](https://github.com/theislab/ehrapy/pull/1120)) @sueoglu
* Update to cookiecutter-scverse v0.8.0, derive the version from git tags via hatch-vcs, and move the `dev` extra to a `dev` dependency group ([#1125](https://github.com/theislab/ehrapy/pull/1125)) @Zethson
* `import ehrapy` no longer loads the holoviews extensions, which now load on the first holoviews-backed plot, and no longer installs a global `SyntaxWarning` filter ([#1129](https://github.com/theislab/ehrapy/pull/1129)) @Zethson
* Drop the unused `thefuzz`, `fhiry` and `filelock` dependencies and move `requests` to the `test` extra ([#1129](https://github.com/theislab/ehrapy/pull/1129)) @Zethson
* Remove dead code and let codecov compare project coverage against the base commit ([#1129](https://github.com/theislab/ehrapy/pull/1129)) @Zethson

## v0.15.0
<!--
anndata 0.13.0 has been released and is now supported.
The anndata 0.13.0 release notes are worth a look: https://anndata.scverse.org/en/stable/release-notes/index.html#v0-13-0
-->

### 🚀 Features

* Add 3D support for {func}`ep.pp.miss_forest_impute <ehrapy.preprocessing.miss_forest_impute>` ([#1052](https://github.com/theislab/ehrapy/pull/1052)) @sueoglu
* Add 3D support for `ep.pp.mice_forest_impute` ([#1055](https://github.com/theislab/ehrapy/pull/1055)) @sueoglu
* Add stratified Table One for baseline comparisons ([#1066](https://github.com/theislab/ehrapy/pull/1066)) @Zethson
* Add CONSORT-style branching for {class}`CohortTracker <ehrapy.tools.CohortTracker>` ([#1077](https://github.com/theislab/ehrapy/pull/1077)) @Zethson
* Add 3D & lazy Dask array support for {func}`ep.pp.encode <ehrapy.preprocessing.encode>` ([#1078](https://github.com/theislab/ehrapy/pull/1078)) @Zethson
* Add in-house causal inference module — {func}`ep.tl.iptw <ehrapy.tools.iptw>`, {func}`ep.tl.g_computation <ehrapy.tools.g_computation>`, {func}`ep.tl.aipw <ehrapy.tools.aipw>`, {func}`ep.tl.propensity_score_matching <ehrapy.tools.propensity_score_matching>`, meta-learners ({func}`ep.tl.t_learner <ehrapy.tools.t_learner>`/{func}`ep.tl.s_learner <ehrapy.tools.s_learner>`/{func}`ep.tl.x_learner <ehrapy.tools.x_learner>`), and {func}`ep.tl.covariate_balance <ehrapy.tools.covariate_balance>` — replacing the `dowhy` dependency ([#1076](https://github.com/theislab/ehrapy/pull/1076)) @Zethson

### 🐛 Bug Fixes

* Sparse array support for `_warn_imputation_threshold` ([#1060](https://github.com/theislab/ehrapy/pull/1060)) @sueoglu

### 🧰 Maintenance

* Adjust to `ehrdata` 0.3.0 release, adding `anndata` 0.13.0 support ([#1084](https://github.com/theislab/ehrapy/pull/1084)) @eroell
* Make `ehrdata` compatible with `anndata>=0.12.13` ([#1065](https://github.com/theislab/ehrapy/pull/1065)) @eroell
* Efficient, dependency-free implementation of the MCAR test ([#1056](https://github.com/theislab/ehrapy/pull/1056)) @agerardy
* Replace `EhrapyConfig` with scverse-misc `Settings` ([#1070](https://github.com/theislab/ehrapy/pull/1070)) @Zethson
* Address pandas `FutureWarning` ([#1063](https://github.com/theislab/ehrapy/pull/1063)) @agerardy
* Remove unused imports, debug prints, and stale TODOs ([#1075](https://github.com/theislab/ehrapy/pull/1075)) @Zethson
* Fix invalid rst roles in `installation.md` leiden section ([#1073](https://github.com/theislab/ehrapy/pull/1073)) @Zethson
* Improve fate tutorial robustness ([#1081](https://github.com/theislab/ehrapy/pull/1081)) @eroell
* Cache CI datasets and harden against flaky downloads ([#1080](https://github.com/theislab/ehrapy/pull/1080)) @sueoglu
* Mirror scanpy's image-comparison tolerances to fix CPU plotting test failures ([#1083](https://github.com/theislab/ehrapy/pull/1083)) @sueoglu
* Unpin numpy in run_notebooks workflow ([#1085](https://github.com/theislab/ehrapy/pull/1085)) @eroell

### 💥 Breaking changes

* Drop AnnData compatibility in favour of an EHRData-only API ([#1069](https://github.com/theislab/ehrapy/pull/1069)) @Zethson
* Drop `leidenalg` flavor from {func}`ep.tl.leiden <ehrapy.tools.leiden>` (igraph only) ([#1072](https://github.com/theislab/ehrapy/pull/1072)) @Zethson


## v0.14.0

### 🚀 Features

* Add LOCF imputation `ep.pp.locf_impute()` for longitudinal (3D) data with forward fill and configurable fallback strategies ([#1020](https://github.com/theislab/ehrapy/pull/1020)) @agerardy @eroell
* Add non-negative CP decomposition `ep.tl.ncp()` for 3D tensor factorisation with companion plots `ep.pl.ncp()` and `ep.pl.ncp_cluster_trajectories()` ([#1030](https://github.com/theislab/ehrapy/pull/1030), [#1038](https://github.com/theislab/ehrapy/pull/1038)) @eroell
* Add `ep.pp.variable_correlations()` and plotting functions `ep.pl.variable_correlations()` / `ep.pl.variable_dependencies()` ([#1010](https://github.com/theislab/ehrapy/pull/1010)) @sueoglu
* Longitudinal explicit impute `ep.pp.explicit_impute()` extended to enable different imputation values per timepoint ([#1023](https://github.com/theislab/ehrapy/pull/1023)) @sueoglu
* Sankey diagram state-transition colours and hover function for timeseries plots ([#1019](https://github.com/theislab/ehrapy/pull/1019)) @sueoglu
* Add longitudinal data analysis notebook ([#1007](https://github.com/theislab/ehrapy/pull/1007)) @eroell
* Add `ep.tl.cox_ph_adjusted_curves()` and plotting functions `ep.pl.cox_ph_adjusted_curves()` ([#1028](https://github.com/theislab/ehrapy/pull/1028)) @sueoglu
* Add 3D support for `ep.pp.knn_impute()` ([#1041](https://github.com/theislab/ehrapy/pull/1041)) @sueoglu
* Add `ep.pp.missing_data_mask()` to compute and apply missing-data masks for sparse, dense, and Dask arrays ([#1045](https://github.com/theislab/ehrapy/pull/1045)) @haoyu-haoyu

### 🧰 Maintenance

* Fix plotting CI ([#1011](https://github.com/theislab/ehrapy/pull/1011)) @sueoglu @eroell
* Continuous values don't repeat title in CohortTracker's barplot ([#1021](https://github.com/theislab/ehrapy/pull/1021)) @sueoglu
* Remove legacy code deprecated in 0.13.0, fix test warnings & adjust to future scanpy arguments ([#1016](https://github.com/theislab/ehrapy/pull/1016)) @eroell
* Update plotting ci dotplot ([#1011](https://github.com/theislab/ehrapy/pull/1011)) @sueoglu @Zethson @eroell
* Add ehrdata as submodule ([#1040](https://github.com/theislab/ehrapy/pull/1040)) @eroell
* Adapt to updated `ed.infer_feature_type`, discard alternative feature type inference in imputation methods ([#1039](https://github.com/theislab/ehrapy/pull/1039)) @eroell
* Upgrade tutorials ([#1042](https://github.com/theislab/ehrapy/pull/1042), [#1043](https://github.com/theislab/ehrapy/pull/1043), [#1046](https://github.com/theislab/ehrapy/pull/1046)) @agerardy @sueoglu @Zethson
* Remove ML with ehrapy notebook ([#1048](https://github.com/theislab/ehrapy/pull/1048)) @eroell

### 🐛 Bug Fixes

* `ep.pp` normalization functions now work when using a `layer` and `.X` is `None` ([#1015](https://github.com/theislab/ehrapy/pull/1015)) @agerardy @eroell
* `ep.tl.rank_features_groups` can use `.obs` regardless of what is in `.X` or `.layers` ([#1015](https://github.com/theislab/ehrapy/pull/1015)) @agerardy @eroell


### ⚠️ Modified
* Update `qc_lab_metrics`([#1025](https://github.com/theislab/ehrapy/pull/1025)) @eroell
* remove deprecated `ep.ad` (moved to `ehrdata`): `infer_feature_types`, `feature_type_overview`, `replace_feature_types`, `anndata_to_df`, `df_to_anndata`, `move_to_obs`, `move_to_x` ([#1016](https://github.com/theislab/ehrapy/pull/1016)) @eroell
* remove deprecated `ep.dt` (moved to `ehrdata`) ([#1016](https://github.com/theislab/ehrapy/pull/1016)) @eroell
* remove deprecated `ep.io` (moved to `ehrdata`): `df_to_anndata`, `read_csv`, `read_fhir`, `read_h5ad`, `write` ([#1016](https://github.com/theislab/ehrapy/pull/1016)) @eroell


## v0.13.1

### 🧰 Maintenance

* improve syntax usage ([#1005](https://github.com/theislab/ehrapy/pull/1005)) @Zethson
* fix fknni extra ([#1003](https://github.com/theislab/ehrapy/pull/1003)) @Zethson

## v0.13.0

### 🚀 Features

* Transitioning from AnnData to EHRData
`EHRData` replaces `AnnData` as ehrapy's core data structure to better support time-series electronic health record data.
The key enhancement is native support for 3D tensors (observations × variables × timesteps) alongside the existing 2D matrices, enabling efficient storage of longitudinal patient data.
A new `.tem` DataFrame provides time-point annotations, complementing the existing `.obs` and `.var` annotations for comprehensive temporal data description.
While `EHRData` maintains full backward compatibility with AnnData's API, users can now seamlessly work with time-series data and leverage specialized methods for temporal analysis.
Existing code using `AnnData` objects will continue to work, but migration to `EHRData` is strongly recommended to access enhanced time-series functionality.
* The preferred central data object is now `EHRData` ([#908](https://github.com/theislab/ehrapy/pull/908)) @eroell
* The `layers` argument is now available for all functions operating on X or layers ([#908](https://github.com/theislab/ehrapy/pull/908)) @eroell
* Update expected behaviour of `io.read_fhir` ([#922](https://github.com/theislab/ehrapy/pull/922)) @eroell
* Move `mimic_2`, `mimic_2_preprocessed`, `diabetes_130_raw`, `diabetes_130_fairlearn` to `ehrdata.dt` ([#908](https://github.com/theislab/ehrapy/pull/908))
* Deprecate all `ep.dt.*`, refer to datasets in `ehrdata` ([#908](https://github.com/theislab/ehrapy/pull/908)) @eroell
* Support Python 3.14 ([#996](https://github.com/theislab/ehrapy/pull/996)) @Zethson
* Move kaplan_meier & cox_ph plots to holoviews ([#995](https://github.com/theislab/ehrapy/pull/995)) @Zethson
* Longitudinal normalization ([#958](https://github.com/theislab/ehrapy/pull/958)) @agerardy
* Add interactive `ols` plot ([#992](https://github.com/theislab/ehrapy/pull/992)) @Zethson
* Longitudinal and new qc_metrics ([#967](https://github.com/theislab/ehrapy/pull/967)) @sueoglu
* Simple Impute for timeseries ([#975](https://github.com/theislab/ehrapy/pull/975)) @eroell
* Simple implementation of balanced sampling ([#937](https://github.com/theislab/ehrapy/pull/937)) @sueoglu
* Add Sankey diagram visualization functions ([#989](https://github.com/theislab/ehrapy/pull/989)) @sueoglu
* Add `ep.pl.timeseries()` to visualize variables over time ([#994](https://github.com/theislab/ehrapy/pull/994)) @sueoglu
* Add GPU CI & skeleton ([#998](https://github.com/theislab/ehrapy/pull/998)) @Zethson
* Add FAMD ([#976](https://github.com/theislab/ehrapy/pull/976)) @Zethson
* 3D enabled implementation of ep.pp.filter_observations, ep.pp.filter_features ([#953](https://github.com/theislab/ehrapy/pull/953)) @sueoglu
* Add time series distances ([#954](https://github.com/theislab/ehrapy/pull/954)) @Zethson

### 🐛 Bug Fixes

* All green if GPU skipped ([#1000](https://github.com/theislab/ehrapy/pull/1000)) @Zethson
* Fix neighbors with timeseries ([#973](https://github.com/theislab/ehrapy/pull/973)) @eroell
* Fix use_rep when X none ([#969](https://github.com/theislab/ehrapy/pull/969)) @eroell
* Fix missing_values_barplot errors ([#963](https://github.com/theislab/ehrapy/pull/963)) @sueoglu
* Fix CR notebook ([#939](https://github.com/theislab/ehrapy/pull/939)) @Zethson

### 🧰 Maintenance

* Update actions ([#977](https://github.com/theislab/ehrapy/pull/977)) @Zethson
* Cleanup simple_impute tests ([#974](https://github.com/theislab/ehrapy/pull/974)) @eroell
* Move to ehrdata 0.0.10 ([#971](https://github.com/theislab/ehrapy/pull/971)) @eroell
* Improved notebook CI ([#959](https://github.com/theislab/ehrapy/pull/959)) @Zethson
* Switch to template ([#960](https://github.com/theislab/ehrapy/pull/960)) @Zethson
* Tests for more plots ([#919](https://github.com/theislab/ehrapy/pull/919)) @sueoglu
* Lowerbound cvxpy ([#935](https://github.com/theislab/ehrapy/pull/935)) @Zethson
* Optimize var_metrics ([#927](https://github.com/theislab/ehrapy/pull/927)) @Zethson
* Refactor Dask usage pattern ([#926](https://github.com/theislab/ehrapy/pull/926)) @Zethson
* Add cover to README & remove some tokens ([#923](https://github.com/theislab/ehrapy/pull/923)) @Zethson
* Update test coverage reporting ([#918](https://github.com/theislab/ehrapy/pull/918)) @eroell
* Fix changelog links ([#915](https://github.com/theislab/ehrapy/pull/915)) @Zethson
* Fixed structure of Returns in _rank_features_groups.py documentation ([#911](https://github.com/theislab/ehrapy/pull/911)) @agerardy
* Add EHRData transition code ([#897](https://github.com/theislab/ehrapy/pull/897)) @Zethson @eroell
* Make test that downloads dermatology dataset more robust ([#906](https://github.com/theislab/ehrapy/pull/906)) @Zethson
* Update image source in README.md ([#986](https://github.com/theislab/ehrapy/pull/986)) @eroell
* Fix plot docs formatting ([#952](https://github.com/theislab/ehrapy/pull/952)) @Zethson
* Typo in the documentation of ehrapy.data.mimic_2_preprocessed ([#917](https://github.com/theislab/ehrapy/pull/917)) @sueoglu

## v0.12.1

### 🚀 Features

* Make dowhy optional & remove medcat ([#903](https://github.com/theislab/ehrapy/pull/903)) @Zethson
* Add about page & improve citations ([#902](https://github.com/theislab/ehrapy/pull/902)) @Zethson
* Overhaul doc structure ([#895](https://github.com/theislab/ehrapy/pull/895)) @Zethson
* Move to biome & improve CI & reenable CR ([#890](https://github.com/theislab/ehrapy/pull/890)) @Zethson
* Clean up Round - cut down anndata extension functionality ([#880](https://github.com/theislab/ehrapy/pull/880)) @eroell

## v0.12.0

### 🚀 Features

* Improved KM plot data depth and functionality ([#853](https://github.com/theislab/ehrapy/pull/853)) @aGuyLearning
* New Feature: Forestplot for CoxPH model ([#838](https://github.com/theislab/ehrapy/pull/838)) @aGuyLearning
* Datatype Support in Quality Control and Impute ([#865](https://github.com/theislab/ehrapy/pull/865)) @aGuyLearning
* Revamp survival analysis interface ([#842](https://github.com/theislab/ehrapy/pull/842)) @aGuyLearning
* Improve submodule documentation ([#859](https://github.com/theislab/ehrapy/pull/859)) @Zethson
* Update Kaplan Meier plots in survival analysis notebook ([#864](https://github.com/theislab/ehrapy/pull/864)) @aGuyLearning

### 🐛 Bug Fixes

* Pass all non-nan features along desired var_names to impute (KNN) ([#867](https://github.com/theislab/ehrapy/pull/867)) @nicolassidoux
* Remove Syntax warnings ([#869](https://github.com/theislab/ehrapy/pull/869)) @Zethson
* Fix test_norm_power_group ([#862](https://github.com/theislab/ehrapy/pull/862)) @Zethson

### 🧰 Maintenance

* Fix a typo in `pl.paga_compare`: `pos` -> `pos,` ([#846](https://github.com/theislab/ehrapy/pull/846)) @VladimirShitov

## v0.11.0

### ✨ Features

* Add array type handling for normalization ([#835](https://github.com/theislab/ehrapy/pull/835)) @eroell @Zethson

### 🐛 Bug Fixes

* Fix scipy array support ([#844](https://github.com/theislab/ehrapy/pull/844)) @Zethson
* Fix casting to float when assigning numeric values; fixes normalization of integer arrays ([#837](https://github.com/theislab/ehrapy/pull/837)) @eroell

## v0.9.0 & 0.10.0

### 🚀 Features

* Make all imputation methods consistent in regard to encoding requirements ([#827](https://github.com/theislab/ehrapy/pull/827)) @nicolassidoux
* Add approximate KNN backend ([#791](https://github.com/theislab/ehrapy/pull/791)) @nicolassidoux
* Improve survival analysis interface ([#825](https://github.com/theislab/ehrapy/pull/825)) @aGuyLearning
* Python 3.12 support ([#794](https://github.com/theislab/ehrapy/pull/794)) @Lilly-May
* Python 3.10+ & use uv for docs & fix RTD & support numpy 2 ([#830](https://github.com/theislab/ehrapy/pull/830)) @Zethson

### 🐛 Bug Fixes

* move_to_x: Fix name of non-implemented argument "copy" to "copy_x", implement & test ([#832](https://github.com/theislab/ehrapy/pull/832)) @eroell
* Contributing typo fix ([#821](https://github.com/theislab/ehrapy/pull/821)) @aGuyLearning
* Fix miceforest ([#800](https://github.com/theislab/ehrapy/pull/800)) @Zethson
* style: == to is for type comparison ([#774](https://github.com/theislab/ehrapy/pull/774)) @eroell

## v0.8.0

### 🚀 Features

* remove pyyaml & explicit scikit-learn ([#729](https://github.com/theislab/ehrapy/pull/729)) @Zethson
* Remove fancyimpute ([#728](https://github.com/theislab/ehrapy/pull/728)) @Zethson
* Unify feature type detection ([#724](https://github.com/theislab/ehrapy/pull/724)) @Lilly-May
* catplot ([#721](https://github.com/theislab/ehrapy/pull/721)) @eroell
* Simplify ehrapy ([#719](https://github.com/theislab/ehrapy/pull/719)) @Zethson
* Use __all__ ([#715](https://github.com/theislab/ehrapy/pull/715)) @Zethson
* Add bias detection to preprocessing ([#690](https://github.com/theislab/ehrapy/pull/690)) @Lilly-May
* Use lamin logger ([#707](https://github.com/theislab/ehrapy/pull/707)) @Zethson
* Add faiss backend for KNN imputation ([#704](https://github.com/theislab/ehrapy/pull/704)) @Zethson
* Build RTD docs with uv ([#700](https://github.com/theislab/ehrapy/pull/700)) @Zethson
* Refactor feature importance ranking ([#698](https://github.com/theislab/ehrapy/pull/698)) @Zethson
* Simplify CI ([#694](https://github.com/theislab/ehrapy/pull/694)) @Zethson
* Refactor outliers and IQR ([#692](https://github.com/theislab/ehrapy/pull/692)) @Zethson
* Calculation of feature importances in a supervised setting ([#677](https://github.com/theislab/ehrapy/pull/677)) @Lilly-May
* Speed up winsorize ([#681](https://github.com/theislab/ehrapy/pull/681)) @Zethson
* Remove notebook prefix in tutorial URLs ([#679](https://github.com/theislab/ehrapy/pull/679)) @Zethson
* Add cohort tracking notebook ([#678](https://github.com/theislab/ehrapy/pull/678)) @Zethson
* Switch to uv ([#674](https://github.com/theislab/ehrapy/pull/674)) @Zethson
* Style: typing of _scale_func_group ([#727](https://github.com/theislab/ehrapy/pull/727)) @eroell
* Improved support of encoded features in detect_bias ([#725](https://github.com/theislab/ehrapy/pull/725)) @Lilly-May
* Enable Synchronous dataloader write ([#722](https://github.com/theislab/ehrapy/pull/722)) @wxicu
* Feature scaling on training set when computing feature importances ([#716](https://github.com/theislab/ehrapy/pull/716)) @Lilly-May
* add batch-wise normalization argument ([#711](https://github.com/theislab/ehrapy/pull/711)) @eroell
* add functools.wraps to type check ([#705](https://github.com/theislab/ehrapy/pull/705)) @eroell
* add bias notebook to list of notebooks ([#696](https://github.com/theislab/ehrapy/pull/696)) @eroell
* basic sampling ([#686](https://github.com/theislab/ehrapy/pull/686)) @eroell
* add options for subitles in legend of cohorttrackers barplot ([#688](https://github.com/theislab/ehrapy/pull/688)) @eroell
* doc fix imputation: 70 instead of 30 ([#683](https://github.com/theislab/ehrapy/pull/683)) @eroell

### 🐛 Bug Fixes

* Encoded dtype to float32 instead of np.number ([#714](https://github.com/theislab/ehrapy/pull/714)) @Zethson
* Fix feature importance warnings ([#708](https://github.com/theislab/ehrapy/pull/708)) @Zethson
* Remove notebook prefix in tutorial URLs ([#679](https://github.com/theislab/ehrapy/pull/679)) @Zethson
* fix name of log_rogistic_aft to log_logistic_aft ([#676](https://github.com/theislab/ehrapy/pull/676)) @eroell

### 🧰 Maintenance

* Remove notebook prefix in tutorial URLs ([#679](https://github.com/theislab/ehrapy/pull/679)) @Zethson
* Add cohort tracking notebook ([#678](https://github.com/theislab/ehrapy/pull/678)) @Zethson
* knni amendments ([#706](https://github.com/theislab/ehrapy/pull/706)) @eroell

## v0.7.0

### 🚀 Features

* Cohort Tracker ([#658](https://github.com/theislab/ehrapy/pull/658)) @eroell
* change diabetes-130 datasets which are provided ([#672](https://github.com/theislab/ehrapy/pull/672)) @eroell
* More sa functions ([#664](https://github.com/theislab/ehrapy/pull/664)) @fatisati
* Coxphfitter ([#643](https://github.com/theislab/ehrapy/pull/643)) @fatisati
* Implement little's test ([#667](https://github.com/theislab/ehrapy/pull/667)) @Zethson
* Improve test design ([#651](https://github.com/theislab/ehrapy/pull/651)) @Zethson
* Improve QC docstring ([#639](https://github.com/theislab/ehrapy/pull/639)) @Zethson
* Refactor _missing_values calculation ([#638](https://github.com/theislab/ehrapy/pull/638)) @Zethson

### 🐛 Bug Fixes

* Fix one-hot encoding tests ([#644](https://github.com/theislab/ehrapy/pull/644)) @Zethson

## v0.6.0

### 🚀 Features

#### Breaking changes

* Move information on numerical/non_numerical/encoded_non_numerical from .uns to .var ([#630](https://github.com/theislab/ehrapy/pull/630)) @eroell

Make older AnnData objects compatible using

```python
def move_type_info_from_uns_to_var(adata, copy=False):
    """Move type information from adata.uns to adata.var['ehrapy_column_type'].

    The latter is the current, updated flavor used by ehrapy.
    """
    if copy:
        adata = adata.copy()

    adata.var["ehrapy_column_type"] = "unknown"

    if "numerical_columns" in adata.uns.keys():
        for key in adata.uns["numerical_columns"]:
            adata.var.loc[key, "ehrapy_column_type"] = "numeric"
    if "non_numerical_columns" in adata.uns.keys():
        for key in adata.uns["non_numerical_columns"]:
            adata.var.loc[key, "ehrapy_column_type"] = "non_numeric"
    if "encoded_non_numerical_columns" in adata.uns.keys():
        for key in adata.uns["encoded_non_numerical_columns"]:
            adata.var.loc[key, "ehrapy_column_type"] = "non_numeric_encoded"

    if copy:
        return adata
```

#### New features

* Medcat refresh ([#623](https://github.com/theislab/ehrapy/pull/623)) @eroell
* Rank features groups obs ([#622](https://github.com/theislab/ehrapy/pull/622)) @eroell
* Add FHIR tutorial and simplify code ([#626](https://github.com/theislab/ehrapy/pull/626)) @Zethson
* Add input checks for imputers ([#625](https://github.com/theislab/ehrapy/pull/625)) @Zethson
* Removed unused dependencies ([#615](https://github.com/theislab/ehrapy/pull/615)) @Zethson
* Refactor encoding ([#588](https://github.com/theislab/ehrapy/pull/588)) @Zethson

### 🐛 Bug Fixes

* Use fixtures for preprocessing tests ([#577](https://github.com/theislab/ehrapy/pull/577)) @Zethson

### 🧰 Maintenance

* Refactoring ([#627](https://github.com/theislab/ehrapy/pull/627)) @Zethson
* Add FHIR tutorial and simplify code ([#626](https://github.com/theislab/ehrapy/pull/626)) @Zethson
* pre-commit ([#587](https://github.com/theislab/ehrapy/pull/587)) @Zethson
* Small edits ([#599](https://github.com/theislab/ehrapy/pull/599)) @eroell

## v0.5.0

### 🚀 Features

* Add g-tests for rank features group ([#546](https://github.com/theislab/ehrapy/pull/546)) @VladimirShitov
* Causal Inference with dowhy ([#502](https://github.com/theislab/ehrapy/pull/502)) @timtreis
* Remove MuData support ([#545](https://github.com/theislab/ehrapy/pull/545)) @Zethson

### 🐛 Bug Fixes

* Fixed reading format warnings  ([#569](https://github.com/theislab/ehrapy/pull/569)) @namsaraeva
* Fixed inability to normalize AnnData that does not require encoding  ([#568](https://github.com/theislab/ehrapy/pull/568)) @namsaraeva
* Fixed adata.uns["non_numericlal_columns"] being empty in mimic_2 dataset ([#567](https://github.com/theislab/ehrapy/pull/567)) @namsaraeva

## v0.4.0

### 🚀 Features

* Add Synthea dataset ([#510](https://github.com/theislab/ehrapy/pull/510)) @namsaraeva
* Added tiny examples to every function ([#498](https://github.com/theislab/ehrapy/pull/498)) @namsaraeva
* add a title parameter ([#494](https://github.com/theislab/ehrapy/pull/494)) @xinyuejohn
* Changed the hue of grey ([#493](https://github.com/theislab/ehrapy/pull/493)) @namsaraeva
* Logger info message when writing to .h5ad files ([#458](https://github.com/theislab/ehrapy/pull/458)) @namsaraeva
* Modified docstrings ([#533](https://github.com/theislab/ehrapy/pull/533)) @namsaraeva
* Added examples to missing modules ([#531](https://github.com/theislab/ehrapy/pull/531)) @namsaraeva
* Allow Python 3.11 ([#523](https://github.com/theislab/ehrapy/pull/523)) @Zethson
* Add test_kmf_logrank ([#516](https://github.com/theislab/ehrapy/pull/516)) @Zethson
* Add scget functions ([#484](https://github.com/theislab/ehrapy/pull/484)) @Zethson
* Add FHIR parsing support ([#463](https://github.com/theislab/ehrapy/pull/463)) @Zethson
* Add new tutorial & switch to python 3.10 ([#454](https://github.com/theislab/ehrapy/pull/454)) @Zethson
* Add docs group ([#437](https://github.com/theislab/ehrapy/pull/437)) @Zethson
* Add thefuzz ([#434](https://github.com/theislab/ehrapy/pull/434)) @Zethson

### 🐛 Bug Fixes

* Fix CI ([#524](https://github.com/theislab/ehrapy/pull/524)) @Zethson
* Error message and minor fixes, issue #447 ([#504](https://github.com/theislab/ehrapy/pull/504)) @namsaraeva
* fix quality control ([#495](https://github.com/theislab/ehrapy/pull/495)) @xinyuejohn
* Fix MacOS CI ([#435](https://github.com/theislab/ehrapy/pull/435)) @Zethson

### 🧰 Maintenance

* Add test_kmf_logrank ([#516](https://github.com/theislab/ehrapy/pull/516)) @Zethson
* Add scget functions ([#484](https://github.com/theislab/ehrapy/pull/484)) @Zethson
* Add new tutorial & switch to python 3.10 ([#454](https://github.com/theislab/ehrapy/pull/454)) @Zethson

## v0.3.0

### 🚀 Features

* Add winsorize, clip quantiles and filter quantiles ([#418](https://github.com/theislab/ehrapy/pull/418)) @Zethson
* Remove PDF support ([#430](https://github.com/theislab/ehrapy/pull/430)) @Zethson
* Logging instance, issue #246 ([#426](https://github.com/theislab/ehrapy/pull/426)) @namsaraeva
* Negative values offset ([#420](https://github.com/theislab/ehrapy/pull/420)) @Zethson
* Missing values visualization, ref issue #271 ([#419](https://github.com/theislab/ehrapy/pull/419)) @namsaraeva
* Add copy_obs parameter to move_to_obs ([#404](https://github.com/theislab/ehrapy/pull/404)) @namsaraeva
* add anova_glm function ([#400](https://github.com/theislab/ehrapy/pull/400)) @xinyuejohn
* issue #397 "check for neighbors run before UMAP" fixed ([#401](https://github.com/theislab/ehrapy/pull/401)) @namsaraeva
* add more tutorials to CI ([#382](https://github.com/theislab/ehrapy/pull/382)) @Zethson
* add support for reading multiple files into Pandas DFs & adapted MIMIC-III Demo ([#386](https://github.com/theislab/ehrapy/pull/386)) @Zethson
* #321: Add X_only option for reading ([#380](https://github.com/theislab/ehrapy/pull/380)) @Imipenem

### 🐛 Bug Fixes

* KeyError fix issue #423 ([#428](https://github.com/theislab/ehrapy/pull/428)) @namsaraeva
* fix qc_metrics bug ([#425](https://github.com/theislab/ehrapy/pull/425)) @xinyuejohn
* df_to_anndata logical XOR to OR, issue #422 ([#429](https://github.com/theislab/ehrapy/pull/429)) @namsaraeva
* Fix docs CI ([#392](https://github.com/theislab/ehrapy/pull/392)) @Zethson
* small fix in the qc_metrics() example ([#407](https://github.com/theislab/ehrapy/pull/407)) @namsaraeva

### 🧰 Maintenance

* Add winsorize, clip quantiles and filter quantiles ([#418](https://github.com/theislab/ehrapy/pull/418)) @Zethson
* Remove PDF support ([#430](https://github.com/theislab/ehrapy/pull/430)) @Zethson
* Negative values offset ([#420](https://github.com/theislab/ehrapy/pull/420)) @Zethson
* Missing values visualization, ref issue #271 ([#419](https://github.com/theislab/ehrapy/pull/419)) @namsaraeva
* Fix docs CI ([#392](https://github.com/theislab/ehrapy/pull/392)) @Zethson

## v0.2.0

### 🚀 Features

* Important cookietemple template update 2.1.0 released! ([#343](https://github.com/theislab/ehrapy/pull/343)) @Zethson
* add chronic kidney disease dataloader ([#301](https://github.com/theislab/ehrapy/pull/301)) @xinyuejohn
* dataloader for diabetes dataset ([#292](https://github.com/theislab/ehrapy/pull/292)) @HorlavaNastassya
* Add X_only option for reading ([#380](https://github.com/theislab/ehrapy/pull/380)) @Imipenem
* MedCAT API improvements & function renaming ([#381](https://github.com/theislab/ehrapy/pull/381)) @Zethson
* minor changes ([#379](https://github.com/theislab/ehrapy/pull/379)) @xinyuejohn
* add functions related to survival analysis ([#371](https://github.com/theislab/ehrapy/pull/371)) @xinyuejohn
* MedCat [#101]: extract biomedical concepts/entities from (free) text ([#367](https://github.com/theislab/ehrapy/pull/367)) @Imipenem
* Add heart dataset to docs ([#377](https://github.com/theislab/ehrapy/pull/377)) @xinyuejohn
* add heart disease data set to ehrapy ([#376](https://github.com/theislab/ehrapy/pull/376)) @xinyuejohn
* add highly_variable_features ([#364](https://github.com/theislab/ehrapy/pull/364)) @xinyuejohn
* add SoftImpute and IterativeSVD to imputation ([#353](https://github.com/theislab/ehrapy/pull/353)) @xinyuejohn
* (#307) Improve KNN with n_neighbours parameter ([#365](https://github.com/theislab/ehrapy/pull/365)) @Imipenem
* add furo theme & switch to markdown ([#359](https://github.com/theislab/ehrapy/pull/359)) @Zethson
* add several datasets and change Docstring examples ([#355](https://github.com/theislab/ehrapy/pull/355)) @xinyuejohn
* Add ability to compare laboratory measurements to reference values ([#352](https://github.com/theislab/ehrapy/pull/352)) @Zethson
* (Feature) New read API #263 ([#351](https://github.com/theislab/ehrapy/pull/351)) @Imipenem
* (Feature) Set index column #305 ([#350](https://github.com/theislab/ehrapy/pull/350)) @Imipenem
* Add encoded parameter to all new datasets amd fix import ([#336](https://github.com/theislab/ehrapy/pull/336)) @xinyuejohn
* (FEATURE) #314: Autodetect binary (0,1) columns ([#327](https://github.com/theislab/ehrapy/pull/327)) @Imipenem
* (FEATURE) Display QC metrics of var #239 ([#323](https://github.com/theislab/ehrapy/pull/323)) @Imipenem
* Add several dataset loaders ([#322](https://github.com/theislab/ehrapy/pull/322)) @xinyuejohn
* (FEATURE) Improve type_overview #306 ([#308](https://github.com/theislab/ehrapy/pull/308)) @Imipenem
* Feature/deep translator integration ([#303](https://github.com/theislab/ehrapy/pull/303)) @MxMstrmn
* remove CLI module ([#298](https://github.com/theislab/ehrapy/pull/298)) @Zethson
* Improve missforest interface ([#284](https://github.com/theislab/ehrapy/pull/284)) @Zethson
* Add example calls and preview images to all plotting functions ([#289](https://github.com/theislab/ehrapy/pull/289)) @xinyuejohn
* add heart failure dataloader ([#291](https://github.com/theislab/ehrapy/pull/291)) @Zethson
* add highly_variable_features ([#364](https://github.com/theislab/ehrapy/pull/364)) @xinyuejohn
* add SoftImpute and IterativeSVD to imputation ([#353](https://github.com/theislab/ehrapy/pull/353)) @xinyuejohn
* add furo theme & switch to markdown ([#359](https://github.com/theislab/ehrapy/pull/359)) @Zethson

### 🐛 Bug Fixes

* (FIX) #255: Encode mutates input adata object ([#348](https://github.com/theislab/ehrapy/pull/348)) @Imipenem
* (FIX) Write .h5ad files ([#347](https://github.com/theislab/ehrapy/pull/347)) @Imipenem
* Fix #331: Improved autodetect docs ([#344](https://github.com/theislab/ehrapy/pull/344)) @Imipenem
* Add encoded parameter to all new datasets amd fix import ([#336](https://github.com/theislab/ehrapy/pull/336)) @xinyuejohn
* (FIX) Autodetect encode + specify encode mode for autodetect ([#310](https://github.com/theislab/ehrapy/pull/310)) @Imipenem

### 🧰 Maintenance

* MedCAT API improvements & function renaming ([#381](https://github.com/theislab/ehrapy/pull/381)) @Zethson
* add functions related to survival analysis ([#371](https://github.com/theislab/ehrapy/pull/371)) @xinyuejohn
* MedCat [#101]: extract biomedical concepts/entities from (free) text ([#367](https://github.com/theislab/ehrapy/pull/367)) @Imipenem
* Add heart dataset to docs ([#377](https://github.com/theislab/ehrapy/pull/377)) @xinyuejohn
* remove CLI module ([#298](https://github.com/theislab/ehrapy/pull/298)) @Zethson

## v0.1.0

### 🚀 Features

* Input and output of CSVs, PDFs, h5ad files
* Several encoding modes (one-hot, label, ...)
* Several imputation methods (simple, KNN, MissForest, ...)
* Several normalization methods (log, scale, ...)
* Full Scanpy API support
* Initial MedCAT integration
* DeepL & Google Translator support
