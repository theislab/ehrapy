from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import scanpy as sc

from ehrapy.anndata import check_feature_types, infer_feature_types, move_to_x
from ehrapy.anndata._constants import (
    CATEGORICAL_TAG,
    DATE_TAG,
    FEATURE_TYPE_KEY,
    NUMERIC_TAG,
)
from ehrapy.preprocessing import encode

if TYPE_CHECKING:
    from anndata import AnnData

    from ehrapy.tools import _method_options


def _merge_arrays(arrays: Iterable[Iterable], groups_order) -> np.recarray:
    """Merge `recarray` obtained from scanpy with manually created numpy `array`"""
    groups_order = list(groups_order)

    # The easiest way to merge recarrays is through dataframe conversion
    dfs = []
    for array in arrays:
        if isinstance(array, np.recarray) or isinstance(array, np.ndarray):
            dfs.append(pd.DataFrame(array, columns=groups_order))
        elif isinstance(array, pd.DataFrame):
            dfs.append(array[groups_order])

    concatenated_arrays = pd.concat(dfs, ignore_index=True, axis=0)

    return concatenated_arrays.to_records(index=False)


def _adjust_pvalues(pvals: np.recarray, corr_method: _method_options._correction_method) -> np.array:
    """Perform per group p-values correction with a given `corr_method`

    Args:
        pvals: numpy records array with p-values. The resulting p-values are corrected per group (i.e. column)
        corr_method: p-value correction method

    Returns:
        Records array of the same format as an input but with corrected p-values
    """
    from statsmodels.stats.multitest import multipletests

    method_map = {"benjamini-hochberg": "fdr_bh", "bonferroni": "bonferroni"}

    pvals_adj = np.ones_like(pvals)

    for group in pvals.dtype.names:
        group_pvals = pvals[group]

        _, group_pvals_adj, _, _ = multipletests(group_pvals, alpha=0.05, method=method_map[corr_method])
        pvals_adj[group] = group_pvals_adj

    return pvals_adj


def _sort_features(adata, key_added="rank_features_groups") -> None:
    """Sort results of :func:`~ehrapy.tl.rank_features_groups` by adjusted p-value

    Args:
        adata: Annotated data matrix after running :func:`~ehrapy.tl.rank_features_groups`
        key_added: The key in `adata.uns` information is saved to.
    """
    if key_added not in adata.uns:
        return

    pvals_adj = adata.uns[key_added]["pvals_adj"]

    for group in pvals_adj.dtype.names:
        group_pvals = pvals_adj[group]
        sorted_indexes = np.argsort(group_pvals)

        for key in adata.uns[key_added].keys():
            if key == "params":
                # This key only stores technical information, nothing to sort here
                continue

            # Sort every key (e.g. pvals, names) by adjusted p-value in an increasing order
            adata.uns[key_added][key][group] = adata.uns[key_added][key][group][sorted_indexes]


def _save_rank_features_result(
    adata, key_added, names, scores, pvals, pvals_adj=None, logfoldchanges=None, pts=None, groups_order=None
) -> None:
    """Write keys with statistical test results to adata.uns

    Args:
        adata: Annotated data matrix after running :func:`~ehrapy.tl.rank_features_groups`
        key_added: The key in `adata.uns` information is saved to.
        names: Structured array storing the feature names
        scores: Array with the statistics
        logfoldchanges: logarithm of fold changes or other info to store under logfoldchanges key
        pvals: p-values of a statistical test
        pts: Percentages of cells containing features
        groups_order: order of groups in structured arrays
    """
    fields = (names, scores, pvals, pvals_adj, logfoldchanges, pts)
    field_names = ("names", "scores", "pvals", "pvals_adj", "logfoldchanges", "pts")

    for values, key in zip(fields, field_names):
        if values is None or not len(values):
            continue

        if key not in adata.uns[key_added]:
            adata.uns[key_added][key] = pd.DataFrame(values, columns=groups_order).to_records(index=False)
        else:
            adata.uns[key_added][key] = _merge_arrays([adata.uns[key_added][key], values], groups_order=groups_order)


def _get_groups_order(groups_subset, group_names, reference):
    """Convert `groups` parameter of :func:`~ehrapy.tl.rank_features_groups` to a list of groups

    Args:
        groups_subset: Subset of groups, e.g. [`'g1'`, `'g2'`, `'g3'`], to which comparison
                       shall be restricted, or `'all'` (default), for all groups.
        group_names: list of all available group names
        reference: One of the groups of `'rest'`

    Returns:
        List of groups, subsetted or full

    Examples:
        >>> _get_groups_order(groups_subset="all", group_names=("A", "B", "C"), reference="B")
        ('A', 'B', 'C')
        >>> _get_groups_order(groups_subset=("A", "B"), group_names=("A", "B", "C"), reference="rest")
        ('A', 'B')
        >>> _get_groups_order(groups_subset=("A", "B"), group_names=("A", "B", "C"), reference="C")
        ('A', 'B', 'C')
    """
    if groups_subset == "all":
        groups_order = group_names
    elif isinstance(groups_subset, (str, int)):
        raise ValueError("Specify a sequence of groups")
    else:
        groups_order = list(groups_subset)
        if isinstance(groups_order[0], int):
            groups_order = [str(n) for n in groups_order]
        if reference != "rest" and reference not in groups_order:
            groups_order += [reference]
    if reference != "rest" and reference not in group_names:
        raise ValueError(f"reference = {reference} needs to be one of groupby = {group_names}.")

    return tuple(groups_order)


@check_feature_types
def _evaluate_categorical_features(
    adata,
    groupby,
    group_names,
    groups: Literal["all"] | Iterable[str] = "all",
    reference: str = "rest",
    categorical_method: _method_options._rank_features_groups_cat_method = "g-test",
    pts=False,
):
    """Run statistical test for categorical features.

    Args:
        adata: Annotated data matrix.
        groupby: The key of the observations grouping to consider.
        groups: Subset of groups, e.g. [`'g1'`, `'g2'`, `'g3'`], to which comparison
                shall be restricted, or `'all'` (default), for all groups.
        reference: If `'rest'`, compare each group to the union of the rest of the group.
                   If a group identifier, compare with respect to this group.
        pts: Whether to add 'pts' key to output. Doesn't contain useful information in this case.
        categorical_method: statistical method to calculate differences between categories

    Returns:
        *names*: `np.array`
                  Structured array to be indexed by group id storing the feature names
        *scores*: `np.array`
                  Array to be indexed by group id storing the statistic underlying
                  the computation of a p-value for each feature for each group.
        *logfoldchanges*: `np.array`
                          Always equal to 1 for this function
        *pvals*: `np.array`
                 p-values of a statistical test
        *pts*: `np.array`
                 Always equal to 1 for this function
    """
    from scipy.stats import chi2_contingency

    tests_to_lambdas = {
        "chi-square": 1,
        "g-test": 0,
        "freeman-tukey": -1 / 2,
        "mod-log-likelihood": -1,
        "neyman": -2,
        "cressie-read": 2 / 3,
    }

    categorical_names = []
    categorical_scores = []
    categorical_pvals = []
    categorical_logfoldchanges = []
    categorical_pts = []

    groups_order = _get_groups_order(groups_subset=groups, group_names=group_names, reference=reference)

    groups_values = adata.obs[groupby].to_numpy()
    for feature in adata.var_names[adata.var[FEATURE_TYPE_KEY] == CATEGORICAL_TAG]:
        if feature == groupby or "ehrapycat_" + feature == groupby or feature == "ehrapycat_" + groupby:
            continue

        try:
            feature_values = adata[:, feature].X.flatten().toarray()
        except ValueError as e:
            raise ValueError(f"Feature {feature} is not encoded. Please encode it using `ehrapy.pp.encode`") from e

        pvals = []
        scores = []

        for group in group_names:
            if group not in groups_order:
                continue

            if reference == "rest":
                reference_mask = (groups_values != group) & np.isin(groups_values, groups_order)
                contingency_table = pd.crosstab(feature_values, reference_mask)
            else:
                obs_to_take = np.isin(groups_values, [group, reference])
                reference_mask = groups_values[obs_to_take] == reference
                contingency_table = pd.crosstab(feature_values[obs_to_take], reference_mask)

            score, p_value, _, _ = chi2_contingency(
                contingency_table.values, lambda_=tests_to_lambdas[categorical_method]
            )
            scores.append(score)
            pvals.append(p_value)

        categorical_names.append([feature] * len(group_names))
        categorical_scores.append(scores)
        categorical_pvals.append(pvals)
        # It is not clear, how to interpret logFC or percentages for categorical data
        # For now, leave some values so that plotting and sorting methods work
        categorical_logfoldchanges.append(np.ones(len(group_names)))
        if pts:
            categorical_pts.append(np.ones(len(group_names)))

    return (
        np.array(categorical_names),
        np.array(categorical_scores),
        np.array(categorical_pvals),
        np.array(categorical_logfoldchanges),
        np.array(categorical_pts),
    )


def _check_no_datetime_columns(df):
    datetime_cols = [
        col
        for col in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_timedelta64_dtype(df[col])
    ]
    if datetime_cols:
        raise ValueError(f"Columns with datetime format found: {datetime_cols}")


def _get_intersection(adata_uns, key, selection):
    """Get intersection of adata_uns[key] and selection"""
    if key in adata_uns:
        uns_enc_to_keep = list(set(adata_uns[key]) & set(selection))
    else:
        uns_enc_to_keep = []
    return uns_enc_to_keep


def _check_columns_to_rank_dict(columns_to_rank):
    if isinstance(columns_to_rank, str):
        if columns_to_rank == "all":
            _var_subset = _obs_subset = False
        else:
            raise ValueError("If columns_to_rank is a string, it must be 'all'.")

    elif isinstance(columns_to_rank, dict):
        allowed_keys = {"var_names", "obs_names"}
        for key in columns_to_rank.keys():
            if key not in allowed_keys:
                raise ValueError(
                    f"columns_to_rank dictionary must have only keys 'var_names' and/or 'obs_names', not {key}."
                )
            if not isinstance(key, str):
                raise ValueError(f"columns_to_rank dictionary keys must be strings, not {type(key)}.")

        for key, value in columns_to_rank.items():
            if not isinstance(value, Iterable) or any(not isinstance(item, str) for item in value):
                raise ValueError(f"The value associated with key '{key}' must be an iterable of strings.")

        _var_subset = "var_names" in columns_to_rank.keys()
        _obs_subset = "obs_names" in columns_to_rank.keys()

    else:
        raise ValueError("columns_to_rank must be either 'all' or a dictionary.")

    return _var_subset, _obs_subset


@check_feature_types
def rank_features_groups(
    adata: AnnData,
    groupby: str,
    groups: Literal["all"] | Iterable[str] = "all",
    reference: str = "rest",
    n_features: int | None = None,
    rankby_abs: bool = False,
    pts: bool = False,
    key_added: str | None = "rank_features_groups",
    copy: bool = False,
    num_cols_method: _method_options._rank_features_groups_method = None,
    cat_cols_method: _method_options._rank_features_groups_cat_method = "g-test",
    correction_method: _method_options._correction_method = "benjamini-hochberg",
    tie_correct: bool = False,
    layer: str | None = None,
    field_to_rank: Literal["layer"] | Literal["obs"] | Literal["layer_and_obs"] = "layer",
    columns_to_rank: dict[str, Iterable[str]] | Literal["all"] = "all",
    **kwds,
) -> None:  # pragma: no cover
    """Rank features for characterizing groups.

    Expects logarithmized data.

    Args:
        adata: Annotated data matrix.
        groupby: The key of the observations grouping to consider.
        groups: Subset of groups, e.g. [`'g1'`, `'g2'`, `'g3'`], to which comparison
                shall be restricted, or `'all'` (default), for all groups.
        reference: If `'rest'`, compare each group to the union of the rest of the group.
                   If a group identifier, compare with respect to this group.
        n_features: The number of features that appear in the returned tables. Defaults to all features.
        rankby_abs: Rank genes by the absolute value of the score, not by the score.
                    The returned scores are never the absolute values.
        pts: Compute the fraction of observations containing the features.
        key_added: The key in `adata.uns` information is saved to.
        copy: Whether to return a copy of the AnnData object.
        num_cols_method:  Statistical method to rank numerical features. The default method is `'t-test'`,
                          `'t-test_overestim_var'` overestimates variance of each group,
                          `'wilcoxon'` uses Wilcoxon rank-sum,
                          `'logreg'` uses logistic regression.
        cat_cols_method: Statistical method to calculate differences between categorical features. The default method is `'g-test'`,
                             `'Chi-square'` tests goodness-of-fit test for categorical data,
                             `'Freeman-Tukey'` tests comparing frequency distributions,
                             `'Mod-log-likelihood'` maximum likelihood estimation,
                             `'Neyman'` tests hypotheses using asymptotic theory,
                             `'Cressie-Read'` is a generalized likelihood test,
        correction_method:  p-value correction method.
                            Used only for statistical tests (e.g. doesn't work for "logreg" `num_cols_method`)
        tie_correct: Use tie correction for `'wilcoxon'` scores. Used only for `'wilcoxon'`.
        layer: Key from `adata.layers` whose value will be used to perform tests on.
        field_to_rank: Set to `layer` to rank variables in `adata.X` or `adata.layers[layer]` (default), `obs` to rank `adata.obs`, or `layer_and_obs` to rank both.
                       Layer needs to be None if this is not 'layer'.
        columns_to_rank: Subset of columns to rank. If 'all', all columns are used.
                         If a dictionary, it must have keys 'var_names' and/or 'obs_names' and values must be iterables of strings
                         such as {'var_names': ['glucose'], 'obs_names': ['age', 'height']}.
        **kwds: Are passed to test methods. Currently, this affects only parameters that
                are passed to :class:`sklearn.linear_model.LogisticRegression`.
                For instance, you can pass `penalty='l1'` to try to come up with a
                minimal set of genes that are good predictors (sparse solution meaning few non-zero fitted coefficients).

    Returns:
        *names* structured `np.ndarray` (`.uns['rank_features_groups']`)
                  Structured array to be indexed by group id storing the gene
                  names. Ordered according to scores.
        *scores* structured `np.ndarray` (`.uns['rank_features_groups']`)
                  Structured array to be indexed by group id storing the z-score
                  underlying the computation of a p-value for each gene for each group.
                  Ordered according to scores.
        *logfoldchanges* structured `np.ndarray` (`.uns['rank_features_groups']`)
                          Structured array to be indexed by group id storing the log2
                          fold change for each gene for each group. Ordered according to scores.
                          Only provided if method is 't-test' like.
                          Note: this is an approximation calculated from mean-log values.
        *pvals* structured `np.ndarray` (`.uns['rank_features_groups']`) p-values.
        *pvals_adj* structured `np.ndarray` (`.uns['rank_features_groups']`) Corrected p-values.
        *pts*: `pandas.DataFrame` (`.uns['rank_features_groups']`)
               Fraction of cells expressing the genes for each group.
        *pts_rest* `pandas.DataFrame` (`.uns['rank_features_groups']`)
                    Only if `reference` is set to `'rest'`.
                    Fraction of observations from the union of the rest of each group containing the features.

     Examples:
         >>> import ehrapy as ep
         >>> adata = ep.dt.mimic_2(encoded=False)
         >>> # want to move some metadata to the obs field
         >>> ep.anndata.move_to_obs(adata, to_obs=["service_unit", "service_num", "age", "mort_day_censored"])
         >>> ep.tl.rank_features_groups(adata, "service_unit")
         >>> ep.pl.rank_features_groups(adata)

         >>> import ehrapy as ep
         >>> adata = ep.dt.mimic_2(encoded=False)
         >>> # want to move some metadata to the obs field
         >>> ep.anndata.move_to_obs(adata, to_obs=["service_unit", "service_num", "age", "mort_day_censored"])
         >>> ep.tl.rank_features_groups(
         ...     adata, "service_unit", field_to_rank="obs", columns_to_rank={"obs_names": ["age", "mort_day_censored"]}
         ... )
         >>> ep.pl.rank_features_groups(adata)

         >>> import ehrapy as ep
         >>> adata = ep.dt.mimic_2(encoded=False)
         >>> # want to move some metadata to the obs field
         >>> ep.anndata.move_to_obs(adata, to_obs=["service_unit", "service_num", "age", "mort_day_censored"])
         >>> ep.tl.rank_features_groups(
         ...     adata,
         ...     "service_unit",
         ...     field_to_rank="layer_and_obs",
         ...     columns_to_rank={"var_names": ["copd_flg", "renal_flg"], "obs_names": ["age", "mort_day_censored"]},
         ... )
         >>> ep.pl.rank_features_groups(adata)

    """
    if layer is not None and field_to_rank == "obs":
        raise ValueError("If 'layer' is not None, 'field_to_rank' cannot be 'obs'.")

    if field_to_rank not in ["layer", "obs", "layer_and_obs"]:
        raise ValueError(f"layer must be one of 'layer', 'obs', 'layer_and_obs', not {field_to_rank}")

    # to give better error messages, check if columns_to_rank have valid keys and values here
    _var_subset, _obs_subset = _check_columns_to_rank_dict(columns_to_rank)

    adata = adata.copy() if copy else adata

    # to create a minimal adata object below, grab a reference to X/layer of the original adata,
    # subsetted to the specified columns
    if field_to_rank in ["layer", "layer_and_obs"]:
        # for some reason ruff insists on this type check. columns_to_rank is always a dict with key "var_names" if _var_subset is True
        if _var_subset and isinstance(columns_to_rank, dict):
            X_to_keep = (
                adata[:, columns_to_rank["var_names"]].X
                if layer is None
                else adata[:, columns_to_rank["var_names"]].layers[layer]
            )
            var_to_keep = adata[:, columns_to_rank["var_names"]].var

        else:
            X_to_keep = adata.X if layer is None else adata.layers[layer]
            var_to_keep = adata.var

    else:
        # dummy 1-dimensional X to be used by move_to_x, and removed again afterwards
        X_to_keep = np.zeros((len(adata), 1))
        var_to_keep = pd.DataFrame({"dummy": [0]})

    adata_minimal = sc.AnnData(
        X=X_to_keep,
        obs=adata.obs,
        var=var_to_keep,
    )

    if field_to_rank in ["obs", "layer_and_obs"]:
        # want columns of obs to become variables in X to be able to use rank_features_groups
        # for some reason ruff insists on this type check. columns_to_rank is always a dict with key "obs_names" if _obs_subset is True
        if _obs_subset and isinstance(columns_to_rank, dict):
            obs_to_move = adata.obs[columns_to_rank["obs_names"]].keys()
        else:
            obs_to_move = adata.obs.keys()
        _check_no_datetime_columns(adata.obs[obs_to_move])
        adata_minimal = move_to_x(adata_minimal, list(obs_to_move))

        if field_to_rank == "obs":
            # the 0th column is a dummy of zeros and is meaningless in this case, and needs to be removed
            adata_minimal = adata_minimal[:, 1:]

        # if the feature type is set in adata.obs, we store the respective feature type in adata_minimal.var
        adata_minimal.var[FEATURE_TYPE_KEY] = [
            adata.var[FEATURE_TYPE_KEY].loc[feature]
            if feature not in adata.obs.keys() and FEATURE_TYPE_KEY in adata.var.keys()
            else CATEGORICAL_TAG
            if adata.obs[feature].dtype == "category"
            else DATE_TAG
            if pd.api.types.is_datetime64_any_dtype(adata.obs[feature])
            else NUMERIC_TAG
            if pd.api.types.is_numeric_dtype(adata.obs[feature])
            else None
            for feature in adata_minimal.var_names
        ]
        # we infer the feature type for all features for which adata.obs did not provide information on the type
        infer_feature_types(adata_minimal, output=None)

        adata_minimal = encode(adata_minimal, autodetect=True, encodings="label")
        # this is needed because encode() doesn't add this key if there are no categorical columns to encode
        if "encoded_non_numerical_columns" not in adata_minimal.uns:
            adata_minimal.uns["encoded_non_numerical_columns"] = []

    if layer is not None:
        adata_minimal.layers[layer] = adata_minimal.X

    # save the reference to the original adata, because we will need to access it later
    adata_orig = adata
    adata = adata_minimal

    if not adata.obs[groupby].dtype == "category":
        adata.obs[groupby] = pd.Categorical(adata.obs[groupby])

    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = {
        "groupby": groupby,
        "reference": reference,
        "method": num_cols_method,
        "categorical_method": cat_cols_method,
        "layer": layer,
        "corr_method": correction_method,
    }

    group_names = pd.Categorical(adata.obs[groupby].astype(str)).categories.tolist()

    if list(adata.var_names[adata.var[FEATURE_TYPE_KEY] == NUMERIC_TAG]):
        # Rank numerical features

        # Without copying `numerical_adata` is a view, and code throws an error
        # because of "object" type of .X
        numerical_adata = adata[:, adata.var_names[adata.var[FEATURE_TYPE_KEY] == NUMERIC_TAG]].copy()
        numerical_adata.X = numerical_adata.X.astype(float)

        sc.tl.rank_genes_groups(
            numerical_adata,
            groupby,
            groups=groups,
            reference=reference,
            n_features=n_features,
            rankby_abs=rankby_abs,
            pts=pts,
            key_added=key_added,
            copy=False,
            method=num_cols_method,
            corr_method=correction_method,
            tie_correct=tie_correct,
            layer=layer,
            **kwds,
        )

        # Update adata.uns with numerical result
        _save_rank_features_result(
            adata,
            key_added,
            names=numerical_adata.uns[key_added]["names"],
            scores=numerical_adata.uns[key_added]["scores"],
            pvals=numerical_adata.uns[key_added]["pvals"],
            pvals_adj=numerical_adata.uns[key_added].get("pvals_adj", None),
            logfoldchanges=numerical_adata.uns[key_added].get("logfoldchanges", None),
            pts=numerical_adata.uns[key_added].get("pts", None),
            groups_order=group_names,
        )

    if list(adata.var_names[adata.var[FEATURE_TYPE_KEY] == CATEGORICAL_TAG]):
        (
            categorical_names,
            categorical_scores,
            categorical_pvals,
            categorical_logfoldchanges,
            categorical_pts,
        ) = _evaluate_categorical_features(
            adata=adata,
            groupby=groupby,
            group_names=group_names,
            groups=groups,
            reference=reference,
            categorical_method=cat_cols_method,
        )

        _save_rank_features_result(
            adata,
            key_added,
            names=categorical_names,
            scores=categorical_scores,
            pvals=categorical_pvals,
            pvals_adj=categorical_pvals.copy(),
            logfoldchanges=categorical_logfoldchanges,
            pts=categorical_pts,
            groups_order=group_names,
        )

    # if field_to_rank was obs or layer_and_obs, the adata object we have been working with is adata_minimal
    adata_orig.uns[key_added] = adata.uns[key_added]
    adata = adata_orig

    if "pvals" in adata.uns[key_added]:
        adata.uns[key_added]["pvals_adj"] = _adjust_pvalues(
            adata.uns[key_added]["pvals"], corr_method=correction_method
        )

    # For some reason, pts should be a DataFrame
    if "pts" in adata.uns[key_added]:
        adata.uns[key_added]["pts"] = pd.DataFrame(adata.uns[key_added]["pts"])

    _sort_features(adata, key_added)

    return adata if copy else None


def filter_rank_features_groups(
    adata: AnnData,
    key="rank_features_groups",
    groupby=None,
    key_added="rank_features_groups_filtered",
    min_in_group_fraction=0.25,
    min_fold_change=1,
    max_out_group_fraction=0.5,
) -> None:  # pragma: no cover
    """Filters out features based on fold change and fraction of features containing the feature within and outside the `groupby` categories.

    See :func:`~ehrapy.tl.rank_features_groups`.

    Results are stored in `adata.uns[key_added]`
    (default: 'rank_genes_groups_filtered').

    To preserve the original structure of adata.uns['rank_genes_groups'],
    filtered genes are set to `NaN`.

    Args:
        adata: Annotated data matrix.
        key: Key previously added by :func:`~ehrapy.tl.rank_features_groups`
        groupby: The key of the observations grouping to consider.
        key_added: The key in `adata.uns` information is saved to.
        min_in_group_fraction: Minimum in group fraction (default: 0.25).
        min_fold_change: Miniumum fold change (default: 1).
        max_out_group_fraction: Maximum out group fraction (default: 0.5).

    Returns:
        Same output as :func:`ehrapy.tl.rank_features_groups` but with filtered feature names set to `nan`

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.tl.rank_features_groups(adata, "service_unit")
        >>> ep.pl.rank_features_groups(adata)
    """
    return sc.tl.filter_rank_genes_groups(
        adata=adata,
        key=key,
        groupby=groupby,
        use_raw=False,
        key_added=key_added,
        min_in_group_fraction=min_in_group_fraction,
        min_fold_change=min_fold_change,
        max_out_group_fraction=max_out_group_fraction,
    )
