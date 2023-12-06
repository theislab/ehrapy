from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from ehrapy.anndata import move_to_x
from ehrapy.preprocessing import encode
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


def _evaluate_categorical_features(
    adata,
    groupby,
    group_names,
    groups: Union[Literal["all"], Iterable[str]] = "all",
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

    for feature in adata.uns["encoded_non_numerical_columns"]:
        if feature == groupby or "ehrapycat_" + feature == groupby or feature == "ehrapycat_" + groupby:
            continue

        feature_values = adata[:, feature].X.flatten().toarray()

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


def rank_features_groups(
    adata: AnnData,
    groupby: str,
    groups: Union[Literal["all"], Iterable[str]] = "all",
    reference: str = "rest",
    n_features: Optional[int] = None,
    rankby_abs: bool = False,
    pts: bool = False,
    key_added: Optional[str] = "rank_features_groups",
    copy: bool = False,
    num_cols_method: _method_options._rank_features_groups_method = None,
    cat_cols_method: _method_options._rank_features_groups_cat_method = "g-test",
    correction_method: _method_options._correction_method = "benjamini-hochberg",
    tie_correct: bool = False,
    layer: Optional[str] = None,
    rank_obs_columns: Optional[Union[Iterable[str], str]] = None,
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
        rank_obs_columns: Whether to rank `adata.obs` columns instead of features in `adata.layer`. If `True`, all observation columns are ranked. If list of column names, only those are ranked.
                    layer needs to be None if this is used.
        **kwds: Are passed to test methods. Currently this affects only parameters that
                are passed to :class:`sklearn.linear_model.LogisticRegression`.
                For instance, you can pass `penalty='l1'` to try to come up with a
                minimal set of genes that are good predictors (sparse solution meaning few non-zero fitted coefficients).

    Returns:
        *names*: structured `np.ndarray` (`.uns['rank_features_groups']`)
                  Structured array to be indexed by group id storing the gene
                  names. Ordered according to scores.
        *scores*: structured `np.ndarray` (`.uns['rank_features_groups']`)
                  Structured array to be indexed by group id storing the z-score
                  underlying the computation of a p-value for each gene for each group.
                  Ordered according to scores.
        *logfoldchanges*: structured `np.ndarray` (`.uns['rank_features_groups']`)
                          Structured array to be indexed by group id storing the log2
                          fold change for each gene for each group. Ordered according to scores.
                          Only provided if method is 't-test' like.
                          Note: this is an approximation calculated from mean-log values.
        *pvals*: structured `np.ndarray` (`.uns['rank_features_groups']`) p-values.
        *pvals_adj* : structured `np.ndarray` (`.uns['rank_features_groups']`) Corrected p-values.
        *pts*: `pandas.DataFrame` (`.uns['rank_features_groups']`)
               Fraction of cells expressing the genes for each group.
        *pts_rest*: `pandas.DataFrame` (`.uns['rank_features_groups']`)
                    Only if `reference` is set to `'rest'`.
                    Fraction of observations from the union of the rest of each group containing the features.

     Examples:
         >>> import ehrapy as ep
         >>> adata = ep.dt.mimic_2(encoded=True)
         >>> ep.tl.rank_features_groups(adata, "service_unit")
         >>> ep.pl.rank_features_groups(adata)
    """
    if layer is not None and rank_obs_columns is not None:
        raise ValueError("Only one of 'layer' and 'rank_obs_columns' can be specified.")

    adata = adata.copy() if copy else adata

    if rank_obs_columns is not None:
        # keep reference to original adata, needed if copy=False
        adata_orig = adata
        adata = sc.AnnData(
            X=np.zeros((len(adata), 1)),
            obs=adata.obs.copy(),
            uns={"numerical_columns": [], "non_numerical_columns": [], "encoded_non_numerical_columns": []},
        )

        if isinstance(rank_obs_columns, str):
            if rank_obs_columns == "all":
                rank_obs_columns = adata.obs.keys().to_list()
            else:
                raise ValueError(
                    f"rank_obs_columns should be 'all' or Iterable of column names, not {rank_obs_columns}."
                )

        # consider adata where all columns from obs become the features, and the other features are dropped
        if not all(elem in adata.obs.columns.values for elem in rank_obs_columns):
            raise ValueError(
                f"Columns `{[col for col in rank_obs_columns if col not in adata.obs.columns.values]}` are not in obs."
            )

        # want columns of obs to become variables in X to be able to use rank_features_groups
        adata_with_moved_columns = move_to_x(adata, list(rank_obs_columns))

        # don't want columns that have been in X before, as only consider columns from obs
        columns_to_select = adata_with_moved_columns.var_names.difference(adata.var_names)
        adata_with_moved_columns = adata_with_moved_columns[:, columns_to_select]

        adata_with_moved_columns = encode(adata_with_moved_columns, autodetect=True, encodings="label")

        adata_with_moved_columns.uns[
            "non_numerical_columns"
        ] = []  # this should be empty, as have only numeric and encoded
        adata_with_moved_columns.uns["numerical_columns"] = adata_with_moved_columns.var_names.difference(
            adata_with_moved_columns.uns["encoded_non_numerical_columns"]
        ).to_list()  # this is sensitive to `encode` really detecting what it should
        adata = adata_with_moved_columns

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

    if adata.uns["numerical_columns"]:
        # Rank numerical features

        # Without copying `numerical_adata` is a view, and code throws an error
        # because of "object" type of .X
        numerical_adata = adata[:, adata.uns["numerical_columns"]].copy()
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

    if adata.uns["encoded_non_numerical_columns"]:
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

    # Adjust p values
    if "pvals" in adata.uns[key_added]:
        adata.uns[key_added]["pvals_adj"] = _adjust_pvalues(
            adata.uns[key_added]["pvals"], corr_method=correction_method
        )

    # For some reason, pts should be a DataFrame
    if "pts" in adata.uns[key_added]:
        adata.uns[key_added]["pts"] = pd.DataFrame(adata.uns[key_added]["pts"])

    _sort_features(adata, key_added)

    if rank_obs_columns is not None:
        adata_orig.uns[key_added] = adata.uns[key_added]
        adata = adata_orig

    return adata if copy else None
