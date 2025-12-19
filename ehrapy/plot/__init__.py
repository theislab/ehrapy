import holoviews as hv

from ehrapy.plot._catplot import catplot
from ehrapy.plot._colormaps import Colormaps, LinearSegmentedColormap
from ehrapy.plot._missingno import (
    missing_values_barplot,
    missing_values_dendrogram,
    missing_values_heatmap,
    missing_values_matrix,
)
from ehrapy.plot._sankey import sankey_diagram, sankey_diagram_time
from ehrapy.plot._scanpy_pl_api import (
    clustermap,
    dendrogram,
    diffmap,
    dotplot,
    dpt_groups_pseudotime,
    dpt_timeseries,
    draw_graph,
    embedding,
    embedding_density,
    heatmap,
    matrixplot,
    paga,
    paga_compare,
    paga_path,
    pca,
    pca_loadings,
    pca_overview,
    pca_variance_ratio,
    rank_features_groups,
    rank_features_groups_dotplot,
    rank_features_groups_heatmap,
    rank_features_groups_matrixplot,
    rank_features_groups_stacked_violin,
    rank_features_groups_tracksplot,
    rank_features_groups_violin,
    ranking,
    scatter,
    stacked_violin,
    tracksplot,
    tsne,
    umap,
    violin,
)
from ehrapy.plot._survival_analysis import cox_ph_forestplot, kaplan_meier, ols
from ehrapy.plot._timeseries import timeseries
from ehrapy.plot.causal_inference._dowhy import causal_effect
from ehrapy.plot.feature_ranking._feature_importances import rank_features_supervised

if not hv.Store.renderers:
    hv.extension("bokeh")
