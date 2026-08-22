"""Workflow steering and next-step suggestions for ehrapy MCP."""

from __future__ import annotations

# Static workflow transitions: (canonical_namespace, function) -> list of (call_string, description)
_WORKFLOW_GRAPH: dict[tuple[str, str], list[tuple[str, str]]] = {
    ("demo", "mimic_2"): [
        ("get_edata_snapshot()", "Inspect observations, variables, and data structure"),
        ("run_preprocessing(function='qc_metrics')", "Calculate missingness and quality metrics"),
    ],
    ("demo", "physionet2012"): [
        ("get_edata_snapshot()", "Inspect observations, variables, and data structure"),
        ("run_preprocessing(function='qc_metrics')", "Calculate missingness and quality metrics"),
    ],
    ("io", "read_csv"): [
        ("get_edata_snapshot()", "Inspect observations, variables, and data structure"),
        ("run_preprocessing(function='qc_metrics')", "Calculate missingness and quality metrics"),
    ],
    ("io", "read_h5ed"): [
        ("get_edata_snapshot()", "Inspect observations, variables, and data structure"),
    ],
    ("preprocessing", "qc_metrics"): [
        (
            "run_preprocessing(function='encode', params={'autodetect': True})",
            "Encode categorical columns into numerical representations",
        ),
        ("run_plot(function='missing_values_matrix')", "Visualize missing data distribution patterns"),
    ],
    ("preprocessing", "encode"): [
        ("run_preprocessing(function='knn_impute')", "Impute missing values (or use miss_forest_impute/simple_impute)"),
        ("run_preprocessing(function='pca')", "Perform Principal Component Analysis for dimensionality reduction"),
    ],
    ("preprocessing", "knn_impute"): [
        ("run_preprocessing(function='pca')", "Perform Principal Component Analysis for dimensionality reduction"),
    ],
    ("preprocessing", "miss_forest_impute"): [
        ("run_preprocessing(function='pca')", "Perform Principal Component Analysis for dimensionality reduction"),
    ],
    ("preprocessing", "simple_impute"): [
        ("run_preprocessing(function='pca')", "Perform Principal Component Analysis for dimensionality reduction"),
    ],
    ("preprocessing", "highly_variable_features"): [
        ("run_preprocessing(function='pca')", "Perform PCA on highly variable features"),
    ],
    ("preprocessing", "pca"): [
        ("run_preprocessing(function='neighbors')", "Compute neighborhood graph from PCA embeddings"),
    ],
    ("preprocessing", "neighbors"): [
        ("run_analysis(function='umap')", "Generate UMAP embedding from neighbor graph"),
        ("run_analysis(function='leiden')", "Cluster observations into sub-cohorts via Leiden algorithm"),
    ],
    ("analysis", "umap"): [
        ("run_plot(function='umap')", "Plot UMAP embeddings colored by clinical variables"),
    ],
    ("analysis", "tsne"): [
        ("run_plot(function='tsne')", "Plot t-SNE embeddings colored by clinical variables"),
    ],
    ("analysis", "leiden"): [
        (
            "run_analysis(function='rank_features_groups', params={'groupby': 'leiden'})",
            "Identify top differentiating clinical features per cluster",
        ),
        ("run_plot(function='umap', params={'color': 'leiden'})", "Visualize clusters on UMAP coordinates"),
    ],
    ("analysis", "rank_features_groups"): [
        ("run_get(function='rank_features_groups_df')", "Extract tabular ranking of differential features"),
        ("run_plot(function='rank_features_groups')", "Plot ranked features per cluster"),
    ],
    ("analysis", "kaplan_meier"): [
        ("run_plot(function='kaplan_meier')", "Render Kaplan-Meier survival curves"),
        (
            "run_analysis(function='cox_ph', params={'duration_col': ..., 'event_col': ..., 'covariates': [...]})",
            "Fit Cox proportional hazards regression model (pass explicit covariates; the full "
            "encoded matrix is collinear and fails with a singularity error)",
        ),
    ],
    ("analysis", "cox_ph"): [
        ("run_plot(function='cox_ph_forestplot')", "Render forest plot of hazard ratios and confidence intervals"),
    ],
    ("analysis", "iptw"): [
        (
            "run_analysis(function='covariate_balance', params={'treatment': ..., 'covariates': [...]})",
            "Check post-weighting covariate balance (`treatment` and `covariates` are required)",
        ),
        (
            "run_plot(function='love_plot')",
            "Render Love plot comparing unadjusted vs adjusted balance (PNG export needs the optional `selenium` package)",
        ),
    ],
    ("analysis", "aipw"): [
        (
            "run_analysis(function='covariate_balance', params={'treatment': ..., 'covariates': [...]})",
            "Check post-weighting covariate balance (`treatment` and `covariates` are required)",
        ),
        (
            "run_plot(function='love_plot')",
            "Render Love plot comparing balance (PNG export needs the optional `selenium` package)",
        ),
    ],
    ("analysis", "g_computation"): [
        (
            "run_analysis(function='covariate_balance', params={'treatment': ..., 'covariates': [...]})",
            "Assess causal treatment effect and covariate balance (`treatment` and `covariates` are required)",
        ),
    ],
}

_NAMESPACE_ALIASES = {
    "tools": "analysis",
    "dt": "demo",
}


def get_suggested_next(namespace: str, function: str) -> list[dict[str, str]] | None:
    """Return suggested next tool calls for a (namespace, function) pair, or None if unknown."""
    canonical_ns = _NAMESPACE_ALIASES.get(namespace, namespace)
    suggestions = _WORKFLOW_GRAPH.get((canonical_ns, function))
    if not suggestions:
        return None
    return [{"call": call, "reason": reason} for call, reason in suggestions[:3]]
