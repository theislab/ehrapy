"""Workflow guides, system prompts, and MCP prompt templates for ehrapy."""

from __future__ import annotations

WORKFLOW_PROMPT = """# ehrapy MCP Server — Agent Guide

## Core Principles
1. **Handle-Based Cohort State:** Datasets are tracked by string identifiers (`edata_id`). Tools implicitly default to the most recently created or modified handle (`used_latest: true`), but explicitly passing `edata_id` is recommended for multi-cohort workflows.
2. **Dual-Channel Output:** Tools return structured metadata in `structured_content` (status, edata_id, dimensions, suggested_next) and formatted Markdown/tables/images in `content`.
3. **Smart Serialization:** Tabular results default to concise column profiles with information ranking. Set `response_format='detailed'` only when you specifically need sample rows.
4. **Non-Destructive Branching:** Use `fork_edata_handle` to preserve intermediate cohort states before irreversible transformations.

## Tool Namespaces
- `preprocessing` (`ep.pp.*`): Quality control (`qc_metrics`), encoding (`encode`), imputation (`knn_impute`, `miss_forest_impute`), normalization (`scale`, `minmax_scale`), PCA (`pca`), neighbor graph (`neighbors`).
- `analysis` (`ep.tl.*`): Survival analysis (`kaplan_meier`, `cox_ph`), causal inference (`iptw`, `aipw`, `g_computation`), embeddings (`umap`, `tsne`), clustering (`leiden`), differential feature ranking (`rank_features_groups`).
- `get` (`ep.get.*`): Read-only data extraction (`obs_df`, `var_df`, `rank_features_groups_df`).
- `plot` (`ep.plot.*`): Visualizations returning PNG image artifacts and file paths.
- `io` (`ehrdata.io.*`): Loading and saving files from/to disk (`read_csv`, `read_h5ed`, `to_pandas`).
- `demo` (`ehrdata.dt.*`): Built-in demonstration datasets (`mimic_2`, `physionet2012`).

## Standard Workflow Sequences

### 1. Exploratory Data Analysis & Quality Control
1. `load_demo_dataset(dataset='mimic_2')` or `ingest_dataset(file_path=...)`
2. `get_edata_snapshot()` → Inspect variables, observations, layers
3. `run_preprocessing(function='qc_metrics')` → Missingness and quality metrics
4. `run_plot(function='missing_values_matrix')` → Visualize missingness patterns

### 2. Dimension Reduction & Subtyping
1. `run_preprocessing(function='encode', params={'autodetect': True})` → Encode categorical features (`autodetect` is required; a bare `encode` call errors)
2. `run_preprocessing(function='knn_impute')` → Impute missing numerical values
3. `run_preprocessing(function='pca')` → Principal component analysis
4. `run_preprocessing(function='neighbors')` → Compute neighborhood graph
5. `run_analysis(function='umap')` → Generate UMAP coordinates
6. `run_analysis(function='leiden')` → Cluster patients into sub-phenotypes
7. `run_plot(function='umap', params={'color': 'leiden'})` → Visualize patient clusters
8. `run_analysis(function='rank_features_groups', params={'groupby': 'leiden'})` → Differentiating features
9. `run_get(function='rank_features_groups_df')` → Tabular differential features

### 3. Survival Analysis
1. `run_analysis(function='kaplan_meier', params={'duration_col': 'mort_day_censored', 'event_col': 'censor_flg', 'groupby': 'service_unit'})`
2. `run_plot(function='kaplan_meier')` → Survival curve plots
3. `run_analysis(function='cox_ph', params={'duration_col': 'mort_day_censored', 'event_col': 'censor_flg', 'covariates': ['age', 'gender_num']})`
4. `run_plot(function='cox_ph_forestplot')` → Hazard ratios forest plot

### 4. Causal Inference
1. `run_analysis(function='iptw', params={'treatment': 'aline_flg', 'outcome': 'hosp_exp_flg', 'covariates': ['age', 'gender_num', 'weight_first']})`
2. `run_analysis(function='covariate_balance')` → Assess balance
3. `run_plot(function='love_plot')` → Render Love plot

## Host vs. Sandbox Filesystem Paths
All file paths passed to `ingest_dataset` or `export_edata` must refer to absolute, host-visible paths. Sandboxed agent paths (e.g. `/workspace`, `/home/claude`) are not accessible directly to the MCP host.
"""

SERVER_INSTRUCTIONS = """ehrapy MCP server provides clinical data analysis tools built on ehrapy and ehrdata.
Operate on clinical cohorts (MIMIC-II, PhysioNet, or custom tabular files) using handle-based EHRData objects.
Always check get_function_help for required parameters before calling unfamiliar preprocessing or analysis functions.
Results are returned with dual-channel structured metadata and compact Markdown tables.
"""
