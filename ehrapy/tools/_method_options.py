from typing import Literal

_Layout = Literal["fr", "drl", "kk", "grid_fr", "lgl", "rt", "rt_circular", "fa"]

_rank_features_groups_method = Literal["logreg", "t-test", "wilcoxon", "t-test_overestim_var"] | None
_correction_method = Literal["benjamini-hochberg", "bonferroni"]
_rank_features_groups_cat_method = Literal[
    "chi-square", "g-test", "freeman-tukey", "mod-log-likelihood", "neyman", "cressie-read"
]
