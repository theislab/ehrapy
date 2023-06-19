from typing import Literal, Optional

_InitPos = Literal["paga", "spectral", "random"]

_LAYOUTS = ("fr", "drl", "kk", "grid_fr", "lgl", "rt", "rt_circular", "fa")
_Layout = Literal[_LAYOUTS]  # type: ignore

_rank_features_groups_method = Optional[Literal["logreg", "t-test", "wilcoxon", "t-test_overestim_var"]]
_correction_method = Literal["benjamini-hochberg", "bonferroni"]
_rank_features_groups_cat_method = Literal[
    "chi-square", "g-test", "freeman-tukey", "mod-log-likelihood", "neyman", "cressie-read"
]

_marker_feature_overlap_methods = Literal["overlap_count", "overlap_coef", "jaccard"]
