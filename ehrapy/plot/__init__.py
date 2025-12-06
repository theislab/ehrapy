import holoviews as hv

from ehrapy.plot._catplot import catplot
from ehrapy.plot._colormaps import Colormaps, LinearSegmentedColormap
from ehrapy.plot._missingno import (
    missing_values_barplot,
    missing_values_dendrogram,
    missing_values_heatmap,
    missing_values_matrix,
)
from ehrapy.plot._scanpy_pl_api import *  # noqa: F403
from ehrapy.plot._survival_analysis import cox_ph_forestplot, kaplan_meier, ols
from ehrapy.plot.causal_inference._dowhy import causal_effect
from ehrapy.plot.feature_ranking._feature_importances import rank_features_supervised

hv.extension("bokeh")
