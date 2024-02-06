from __future__ import annotations

import random
from collections import OrderedDict
from string import ascii_letters
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from anndata import AnnData, concat
from rich import print
from rich.text import Text
from rich.tree import Tree
from scanpy.get import obs_df, rank_genes_groups_df, var_df
from scipy import sparse
from scipy.sparse import issparse

from ehrapy import logging as logg
from ehrapy.anndata._constants import EHRAPY_TYPE_KEY, NON_NUMERIC_ENCODED_TAG, NON_NUMERIC_TAG, NUMERIC_TAG

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence


class BaseDataframes(NamedTuple):
    obs: pd.DataFrame
    df: pd.DataFrame


def df_to_anndata(
    df: pd.DataFrame, columns_obs_only: list[str] | None = None, index_column: str | None = None
) -> AnnData:
    """Transform a given pandas dataframe into an AnnData object.

    Note that columns containing boolean values (either 0/1 or T(t)rue/F(f)alse)
    will be stored as boolean columns whereas the other non numerical columns will be stored as categorical values.

    Args:
        df: The pandas dataframe to be transformed
        columns_obs_only: An optional list of column names that should belong to obs only and not X
        index_column: The index column of obs. This can be either a column name (or its numerical index in the dataframe) or the index of the dataframe

    Returns:
        An AnnData object created from the given pandas dataframe

    Examples:
        >>> import ehrapy as ep
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "patient_id": ["0", "1", "2", "3", "4"],
        ...         "age": [65, 72, 58, 78, 82],
        ...         "sex": ["M", "F", "F", "M", "F"],
        ...     }
        ... )
        >>> adata = ep.df_to_anndata(df, index_column="patient_id")
    """
    # allow index 0
    if index_column is not None:
        df_columns = list(df.columns)
        # if the index of the dataframe is the index_column leave it as it is
        if index_column == df.index.name:
            pass
        # if index column is either numerical or not the actual index name, search for index_column in the columns
        elif isinstance(index_column, int) or index_column != df.index.name:
            if isinstance(index_column, str) and index_column in df_columns:
                df = df.set_index(index_column)
            # also ensure that the index is in range
            elif isinstance(index_column, int) and index_column < len(df_columns):
                df = df.set_index(df_columns[index_column])
            else:
                raise ValueError(f"Did not find column {index_column} in neither index or columns!")
        # index_column is neither in the index or in the columns or passed as some value that could not be understood
        else:  # pragma: no cover
            raise ValueError(f"Did not find column {index_column} in neither index or columns!")

    # move columns from the input dataframe to obs
    if columns_obs_only:
        try:
            obs = df[columns_obs_only].copy()
            obs = obs.set_index(df.index.map(str))
            df = df.drop(columns_obs_only, axis=1)
        except KeyError as e:
            raise ValueError(
                "One or more column names passed to column_obs_only were not found in the input data. "
                "Are the column names spelled correctly?"
            ) from e
    else:
        obs = pd.DataFrame(index=df.index.map(str))
    dataframes = BaseDataframes(obs, df)
    numerical_columns = list(dataframes.df.select_dtypes("number").columns)
    # if data is numerical only, short-circuit AnnData creation to have float dtype instead of object
    X = dataframes.df.to_numpy(copy=True)

    # initializing an OrderedDict with a non-empty dict might not be intended,
    # see: https://stackoverflow.com/questions/25480089/right-way-to-initialize-an-ordereddict-using-its-constructor-such-that-it-retain/25480206
    uns = OrderedDict()  # type: ignore
    # store all numerical/non-numerical columns that are not obs only
    binary_columns = _detect_binary_columns(df, numerical_columns)

    var = pd.DataFrame(index=list(dataframes.df.columns))
    var[EHRAPY_TYPE_KEY] = NON_NUMERIC_TAG
    var.loc[var.index.isin(list(set(numerical_columns) | set(binary_columns))), EHRAPY_TYPE_KEY] = NUMERIC_TAG
    # in case of encoded columns by ehrapy, want to be able to read it back in
    var.loc[var.index.str.contains("ehrapycat"), EHRAPY_TYPE_KEY] = NON_NUMERIC_ENCODED_TAG

    all_num = True if len(numerical_columns) == len(list(dataframes.df.columns)) else False
    X = X.astype(np.number) if all_num else X.astype(object)
    # cast non numerical obs only columns to category or bool dtype, which is needed for writing to .h5ad files
    adata = AnnData(
        X=X,
        obs=_cast_obs_columns(dataframes.obs),
        var=var,
        layers={"original": X.copy()},
        uns=uns,
    )

    logg.info(
        f"Transformed passed DataFrame into an AnnData object with n_obs x n_vars = `{adata.n_obs}` x `{adata.n_vars}`."
    )

    return adata


def anndata_to_df(
    adata: AnnData, layer: str = None, obs_cols: list[str] | str | None = None, var_cols: list[str] | str | None = None
) -> pd.DataFrame:
    """Transform an AnnData object to a pandas DataFrame.

    Args:
        adata: The AnnData object to be transformed into a pandas Dataframe
        layer: The layer to use for X.
        obs_cols: List of obs columns to add to X.
        var_cols: List of var columns to add to X.

    Returns:
        The AnnData object as a pandas Dataframe

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> df = ep.ad.anndata_to_df(adata)
    """
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X
    if issparse(X):  # pragma: no cover
        X = X.toarray()

    df = pd.DataFrame(X, columns=list(adata.var_names))
    if obs_cols:
        if len(adata.obs.columns) == 0:
            raise ValueError("Cannot slice columns from empty obs!")
        if isinstance(obs_cols, str):
            obs_cols = list(obs_cols)
        if isinstance(obs_cols, list):  # pragma: no cover
            obs_slice = adata.obs[obs_cols]
        # reset index needed since we slice all or at least some columns from obs DataFrame
        obs_slice = obs_slice.reset_index(drop=True)
        df = pd.concat([df, obs_slice], axis=1)
        logg.info(f"Added `{obs_cols}` columns to `X`.")
    if var_cols:
        if len(adata.var.columns) == 0:
            raise ValueError("Cannot slice columns from empty var!")
        if isinstance(var_cols, str):
            var_cols = list(var_cols)
        if isinstance(var_cols, list):
            var_slice = adata.var[var_cols]
        # reset index needed since we slice all or at least some columns from var DataFrame
        var_slice = var_slice.reset_index(drop=True)
        df = pd.concat([df, var_slice], axis=1)

    return df


def move_to_obs(adata: AnnData, to_obs: list[str] | str, copy_obs: bool = False) -> AnnData:
    """Move inplace or copy features from X to obs.

    Note that columns containing boolean values (either 0/1 or True(true)/False(false))
    will be stored as boolean columns whereas the other non numerical columns will be stored as categorical.

    Args:
        adata: The AnnData object
        to_obs: The columns to move to obs
        copy_obs: The values are copied to obs (and therefore kept in X) instead of moved completely

    Returns:
        The original AnnData object with moved or copied columns from X to obs

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.ad.move_to_obs(adata, ["age"], copy_obs=False)
    """
    if isinstance(to_obs, str):  # pragma: no cover
        to_obs = [to_obs]

    # don't allow moving encoded columns as this could lead to inconsistent data in X and obs
    if any(column.startswith("ehrapycat") for column in to_obs):
        raise ValueError(
            "Cannot move encoded columns from X to obs. Either undo encoding or remove them from the list!"
        )

    if not all(elem in adata.var_names.values for elem in to_obs):
        raise ValueError(
            f"Columns `{[col for col in to_obs if col not in adata.var_names.values]}` are not in var_names."
        )

    cols_to_obs_indices = adata.var_names.isin(to_obs)

    num_set = _get_var_indices_for_type(adata, NUMERIC_TAG)
    var_num = list(set(to_obs) & set(num_set))

    if copy_obs:
        cols_to_obs = adata[:, cols_to_obs_indices].to_df()
        adata.obs = adata.obs.join(cols_to_obs)
        adata.obs[var_num] = adata.obs[var_num].apply(pd.to_numeric, errors="ignore", downcast="float")
        adata.obs = _cast_obs_columns(adata.obs)
    else:
        df = adata[:, cols_to_obs_indices].to_df()
        adata._inplace_subset_var(~cols_to_obs_indices)
        adata.obs = adata.obs.join(df)
        adata.obs[var_num] = adata.obs[var_num].apply(pd.to_numeric, errors="ignore", downcast="float")
        adata.obs = _cast_obs_columns(adata.obs)

    logg.info(f"Added `{to_obs}` to `obs`.")

    return adata


def _get_var_indices_for_type(adata: AnnData, tag: str) -> list[str]:
    """Get indices of columns in var for a given tag.

    Args:
        adata: The AnnData object
        tag: The tag to search for, should be one of `NUMERIC_TAG`, `NON_NUMERIC_TAG` or `NON_NUMERIC_ENCODED_TAG`

    Returns:
        List of numeric columns
    """
    return adata.var_names[adata.var[EHRAPY_TYPE_KEY] == tag].tolist()


def delete_from_obs(adata: AnnData, to_delete: list[str]) -> AnnData:
    """Delete features from obs.

    Args:
        adata: The AnnData object
        to_delete: The columns to delete from obs

    Returns:
        The original AnnData object with deleted columns from obs.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.ad.move_to_obs(adata, ["age"], copy_obs=True)
        >>> ep.ad.delete_from_obs(adata, ["age"])
    """
    if isinstance(to_delete, str):  # pragma: no cover
        to_delete = [to_delete]

    if not all(elem in adata.obs.columns.values for elem in to_delete):
        raise ValueError(
            f"Columns `{[col for col in to_delete if col not in adata.obs.columns.values]}` are not in obs."
        )

    adata.obs = adata.obs[adata.obs.columns[~adata.obs.columns.isin(to_delete)]]

    logg.info(f"Removed `{to_delete}` from `obs`.")

    return adata


def move_to_x(adata: AnnData, to_x: list[str] | str) -> AnnData:
    """Move features from obs to X inplace.

    Args:
        adata: The AnnData object
        to_x: The columns to move to X
        copy: Whether to return a copy or not

    Returns:
        A new AnnData object with moved columns from obs to X. This should not be used for datetime columns currently.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.ad.move_to_obs(adata, ["age"], copy_obs=False)
        >>> new_adata = ep.ad.move_to_x(adata, ["age"])
    """
    if isinstance(to_x, str):  # pragma: no cover
        to_x = [to_x]

    if not all(elem in adata.obs.columns.values for elem in to_x):
        raise ValueError(f"Columns `{[col for col in to_x if col not in adata.obs.columns.values]}` are not in obs.")

    cols_present_in_x = []
    cols_not_in_x = []

    for col in to_x:
        if col in set(adata.var_names):
            cols_present_in_x.append(col)
        else:
            cols_not_in_x.append(col)

    if cols_present_in_x:
        logg.info(
            f"Columns `{cols_present_in_x}` are already in X. Skipped moving `{cols_present_in_x}` to X. "
            f"If you want to permanently delete these columns from obs, please use the function delete_from_obs()."
        )

    if cols_not_in_x:
        new_adata = concat([adata, AnnData(adata.obs[cols_not_in_x])], axis=1)
        new_adata.obs = adata.obs[adata.obs.columns[~adata.obs.columns.isin(cols_not_in_x)]]

        # AnnData's concat discards var if they dont match in their keys, so we need to create a new var
        created_var = _create_new_var(adata, cols_not_in_x)
        new_adata.var = pd.concat([adata.var, created_var], axis=0)

        logg.info(f"Added `{cols_not_in_x}` features to `X`.")
    else:
        new_adata = adata

    return new_adata


def _get_column_indices(adata: AnnData, col_names: str | Iterable[str]) -> list[int]:
    """Fetches the column indices in X for a given list of column names

    Args:
        adata: :class:`~anndata.AnnData` object.
        col_names: Column names to extract the indices for.

    Returns:
        List of column indices.
    """
    col_names = [col_names] if isinstance(col_names, str) else col_names
    mask = np.isin(adata.var_names, col_names)
    indices = np.where(mask)[0].tolist()

    return indices


def type_overview(data: AnnData, sort_by: str | None = None, sort_reversed: bool = False) -> None:  # pragma: no cover
    """Prints the current state of an :class:`~anndata.AnnData` object in a tree format.

    Output can be printed in sorted format by using one of `dtype`, `order`, `num_cats` or `None`, which sorts by data type, lexicographical order,
    number of unique values (excluding NaN's) and unsorted respectively. Note that sorting by `num_cats` only affects
    encoded variables currently and will display unencoded vars unsorted.

    Args:
        data: :class:`~anndata.AnnData` object to display
        sort_by: How the tree output should be sorted. One of `dtype`, `order`, `num_cats` or None (Defaults to None -> unsorted)
        sort_reversed: Whether to sort in reversed order or not

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.ad.type_overview(adata)
    """
    if isinstance(data, AnnData):
        _adata_type_overview(data, sort_by, sort_reversed)
    else:
        raise ValueError(f"Unable to present object of type {type(data)}. Can only display AnnData objects!")


def _adata_type_overview(
    adata: AnnData, sort_by: str | None = None, sort_reversed: bool = False
) -> None:  # pragma: no cover
    """Display the :class:`~anndata.AnnData object in its current state (encoded and unencoded variables, obs)

    Args:
        adata: The :class:`~anndata.AnnData object to display
        sort_by: Whether to sort output or not
        sort_reversed: Whether to sort output in reversed order or not
    """

    tree = Tree(
        f"[b green]Variable names for AnnData object with {len(adata.obs_names)} obs and {len(adata.var_names)} vars",
        guide_style="underline2 bright_blue",
    )
    if "var_to_encoding" in adata.uns.keys():
        original_values = adata.uns["original_values_categoricals"]
        branch = tree.add("🔐 Encoded variables", style="b green")
        dtype_dict = _infer_dtype_per_encoded_var(list(original_values.keys()), original_values)
        # sort encoded vars by lexicographical order of original values
        if sort_by == "order":
            encoded_list = sorted(original_values.keys(), reverse=sort_reversed)
            for categorical in encoded_list:
                branch.add(
                    f"[blue]{categorical} -> {dtype_dict[categorical][1]} categories;"
                    f" [green]{adata.uns['var_to_encoding'][categorical].replace('encoding', '').replace('_', ' ').strip()} [blue]encoded; [green]original data type: [blue]{dtype_dict[categorical][0]}"
                )
        # sort encoded vars by data type of the original values or the number of unique values in original data (excluding NaNs)
        elif sort_by == "dtype" or sort_by == "num_cats":
            sorted_by_type = dict(
                sorted(
                    dtype_dict.items(), key=lambda item: item[1][0 if sort_by == "dtype" else 1], reverse=sort_reversed
                )
            )
            for categorical in sorted_by_type:
                branch.add(
                    f"[blue]{categorical} -> {sorted_by_type[categorical][1]} categories;"
                    f" [green]{adata.uns['var_to_encoding'][categorical].replace('encoding', '').replace('_', ' ').strip()} [blue]encoded; [green]original data type: [blue]{sorted_by_type[categorical][0]}"
                )
        # display in unsorted order
        else:
            encoded_list = original_values.keys()
            for categorical in encoded_list:
                branch.add(
                    f"[blue]{categorical} -> {dtype_dict[categorical][1]} categories;"
                    f" [green]{adata.uns['var_to_encoding'][categorical].replace('encoding', '').replace('_', ' ').strip()} [blue]encoded; [green]original data type: [blue]{dtype_dict[categorical][0]}"
                )
    branch_num = tree.add(Text("🔓 Unencoded variables"), style="b green")

    if sort_by == "order":
        var_names = sorted(adata.var_names.values, reverse=sort_reversed)
        _sort_by_order_or_none(adata, branch_num, var_names)
    elif sort_by == "dtype":
        var_names = list(adata.var_names.values)
        _sort_by_type(adata, branch_num, var_names, sort_reversed)
    else:
        var_names = list(adata.var_names.values)
        _sort_by_order_or_none(adata, branch_num, var_names)

    if sort_by:
        logg.info(
            "Displaying AnnData object in sorted mode. Note that this might not be the exact same order of the variables in X or var are stored!"
        )
    print(tree)


def _sort_by_order_or_none(adata: AnnData, branch, var_names: list[str]):
    """Add branches to tree for sorting by order or unsorted."""
    var_names_val = list(adata.var_names.values)
    for other_vars in var_names:
        if not other_vars.startswith("ehrapycat"):
            idx = var_names_val.index(other_vars)
            unique_categoricals = pd.unique(adata.X[:, idx : idx + 1].flatten())
            data_type = pd.api.types.infer_dtype(unique_categoricals)
            branch.add(f"[blue]{other_vars} -> [green]data type: [blue]{data_type}")


def _sort_by_type(adata: AnnData, branch, var_names: list[str], sort_reversed: bool):
    """Sort tree output by datatype"""
    tmp_dict = {}
    var_names_val = list(adata.var_names.values)

    for other_vars in var_names:
        if not other_vars.startswith("ehrapycat"):
            idx = var_names_val.index(other_vars)
            unique_categoricals = pd.unique(adata.X[:, idx : idx + 1].flatten())
            data_type = pd.api.types.infer_dtype(unique_categoricals)
            tmp_dict[other_vars] = data_type

    sorted_by_type = dict(sorted(tmp_dict.items(), key=lambda item: item[1], reverse=sort_reversed))
    for var in sorted_by_type:
        branch.add(f"[blue]{var} -> [green]data type: [blue]{sorted_by_type[var]}")


def _infer_dtype_per_encoded_var(encoded_list: list[str], original_values) -> dict[str, tuple[str, int]]:
    """Infer dtype of each encoded varibale of an AnnData object."""
    dtype_dict = {}
    for categorical in encoded_list:
        unique_categoricals = pd.unique(original_values[categorical].flatten())
        categorical_type = pd.api.types.infer_dtype(unique_categoricals)
        num_unique_values = pd.DataFrame(unique_categoricals).dropna()[0].nunique()
        dtype_dict[categorical] = (categorical_type, num_unique_values)

    return dtype_dict


def _single_quote_string(name: str) -> str:  # pragma: no cover
    """Single quote a string to inject it into f-strings, since backslashes cannot be in double f-strings."""
    return f"'{name}'"


def _assert_encoded(adata: AnnData):
    try:
        assert np.issubdtype(adata.X.dtype, np.number)
    except AssertionError:
        raise NotEncodedError("The AnnData object has not yet been encoded.") from AssertionError


def get_numeric_vars(adata: AnnData) -> list[str]:
    """Fetches the column names for numeric variables in X.

    Args:
        adata: :class:`~anndata.AnnData` object

    Returns:
        List of column numeric column names
    """
    _assert_encoded(adata)

    # This behaviour is consistent with the previous behaviour, allowing for a simple fully numeric X
    if EHRAPY_TYPE_KEY not in adata.var.columns:
        return list(adata.var_names.values)
    else:
        return _get_var_indices_for_type(adata, NUMERIC_TAG)


def assert_numeric_vars(adata: AnnData, vars: Sequence[str]):
    num_vars = get_numeric_vars(adata)

    try:
        assert set(vars) <= set(num_vars)
    except AssertionError:
        raise ValueError("Some selected vars are not numeric") from None


def set_numeric_vars(
    adata: AnnData, values: np.ndarray, vars: Sequence[str] | None = None, copy: bool = False
) -> AnnData | None:
    """Sets the numeric values in given column names in X.

    Args:
        adata: :class:`~anndata.AnnData` object
        values: Matrix containing the replacement values
        vars: List of names of the numeric variables to replace. If `None` they will be detected using :func:`~ehrapy.preprocessing.get_numeric_vars`.
        copy: Whether to return a copy with the normalized data.

    Returns:
        :class:`~anndata.AnnData` object with updated X
    """
    _assert_encoded(adata)

    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"Values must be numeric (current dtype is {values.dtype})")

    n_values = values.shape[1]

    if n_values != len(vars):
        raise ValueError(f"Number of values ({n_values}) does not match number of vars ({len(vars)})")

    if copy:
        adata = adata.copy()

    vars_idx = _get_column_indices(adata, vars)

    for i in range(n_values):
        adata.X[:, vars_idx[i]] = values[:, i]

    logg.info(f"Values in columns {vars} were replaced by {values}.")

    return adata


def _update_uns(
    adata: AnnData, moved_columns: list[str], to_x: bool = False
) -> tuple[list[str], list[str], list[str] | None]:
    """Updates .uns of adata to reflect the changes made on the object by moving columns from X to obs or vice versa.

    1.) Moving `col1` from `X` to `obs`: `col1` is either numerical or non_numerical, so delete it from the corresponding entry in `uns`
    2.) Moving `col1` from `obs` to `X`: `col1` is either numerical or non_numerical, so add it to the corresponding entry in `uns`

    Args:
        adata: class:`~anndata.AnnData` object
        moved_columns: List of column names to be moved
        to_x: Whether to move from `obs` to `X` or vice versa

    Returns:
        :class:`~anndata.AnnData` object with updated .uns
    """
    moved_columns_set = set(moved_columns)
    if not to_x:  # moving from `X` to `obs`, delete it from the corresponding entry in `uns`.
        num_set = set(adata.uns["numerical_columns"].copy())
        non_num_set = set(adata.uns["non_numerical_columns"].copy())
        var_num = []
        for var in moved_columns_set:
            if var in num_set:
                var_num.append(var)
                num_set -= {var}
            elif var in non_num_set:
                non_num_set -= {var}
        logg.info(f"Added `{moved_columns}` columns to `X`.")
        return list(num_set), list(non_num_set), var_num
    else:  # moving from `obs` to `X`, add it to the corresponding entry in `uns`.
        all_moved_non_num_columns = moved_columns_set & set(adata.obs.select_dtypes(exclude="number").columns)
        all_moved_num_columns = list(moved_columns_set ^ all_moved_non_num_columns)
        logg.info(f"Added `{moved_columns}` columns to `obs`.")
        return all_moved_num_columns, list(all_moved_non_num_columns), None


def _create_new_var(adata: AnnData, cols_not_in_x: list[str]) -> pd.DataFrame:
    """Create a new var DataFrame with the EHRAPY_TYPE_KEY column set for entries from .obs.

    Args:
        adata: From where to get the .obs
        cols_not_in_x: .obs columns to move to X

    Returns:
        New var DataFrame with EHRAPY_TYPE_KEY column set for entries from .obs
    """
    all_moved_num_columns = set(cols_not_in_x) & set(adata.obs.select_dtypes(include="number").columns)

    new_var = pd.DataFrame(index=cols_not_in_x)
    new_var[EHRAPY_TYPE_KEY] = NON_NUMERIC_TAG
    new_var.loc[list(all_moved_num_columns), EHRAPY_TYPE_KEY] = NUMERIC_TAG

    return new_var


def _detect_binary_columns(df: pd.DataFrame, numerical_columns: list[str]) -> list[str]:
    """Detect all columns that contain only 0 and 1 (besides NaNs).

    Args:
        df: The dataframe to check.
        numerical_columns: All numerical columns of the dataframe.

    Returns:
            List of column names that are binary (containing only 0 and 1 (+NaNs))
    """
    binary_columns = []
    for column in numerical_columns:
        # checking for float and int as well as NaNs (this is safe since checked columns are numericals only)
        # only columns that contain at least one 0 and one 1 are counted as binary (or 0.0/1.0)
        if df[column].isin([0.0, 1.0, np.NaN, 0, 1]).all() and df[column].nunique() == 2:
            binary_columns.append(column)

    return binary_columns


def _cast_obs_columns(obs: pd.DataFrame) -> pd.DataFrame:
    """Cast non numerical obs columns to either category or bool.
    Args:
        obs: Obs of an AnnData object

    Returns:
        The type casted obs.
    """
    # only cast non numerical columns
    object_columns = list(obs.select_dtypes(exclude=["number", "category", "bool"]).columns)
    # type cast each non numerical column to either bool (if possible) or category else
    obs[object_columns] = obs[object_columns].apply(
        lambda obs_name: obs_name.astype("category")
        if not set(pd.unique(obs_name)).issubset({False, True, np.NaN})
        else obs_name.astype("bool"),
        axis=0,
    )
    return obs


def generate_anndata(  # pragma: no cover
    shape: tuple[int, int],
    X_type=sparse.csr_matrix,
    X_dtype=np.float32,
    obsm_types: Collection = (sparse.csr_matrix, np.ndarray, pd.DataFrame),
    varm_types: Collection = (sparse.csr_matrix, np.ndarray, pd.DataFrame),
    layers_types: Collection = (sparse.csr_matrix, np.ndarray, pd.DataFrame),
    include_nlp: bool = False,
) -> AnnData:
    """Generates a predefined AnnData with random values.

    Args:
        shape: Shape of the X matrix.
        X_type: Type of the X matrix.
        X_dtype: Data type of the X matrix.
        obsm_types: Types of the obsm matrices.
        varm_types: Types of the varm matrices.
        layers_types: Types of additional layers.
        include_nlp: Whether to include diseases for NLP in all of X, obs and var.
                     Sets the X_dtype to object by default and overwrites the passed X_dtype.

    Returns:
        A specified AnnData object.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.ad.generate_anndata((2, 2), include_nlp=True)
    """
    example_diseases: list[str] = ["diabetes melitus", "breast cancer", "dementia", "pneumonia"]

    M, N = shape
    obs_names = pd.Index(f"patient{i}" for i in range(shape[0]))
    var_names = pd.Index(f"feature{i}" for i in range(shape[1]))

    def _generate_typed_df(n_values, index=None, nlp: bool = False) -> pd.DataFrame:
        """Generates a typed DataFrame with categoricals and numericals.

        Args:
            n_values: Number of values to generate per type.
            index: Name of the index column.
            nlp: Whether to include disease names.

        Returns:
            Pandas DataFrame with the specified number of values.
        """
        letters = np.fromiter(iter(ascii_letters), "U1")
        if n_values > len(letters):
            letters = letters[: n_values // 2]  # Make sure categories are repeated
        df = pd.DataFrame(
            {
                "cat": pd.Categorical(np.random.choice(letters, n_values)),
                "cat_ordered": pd.Categorical(np.random.choice(letters, n_values), ordered=True),
                "int64": np.random.randint(-50, 50, n_values),
                "float64": np.random.random(n_values),
                "uint8": np.random.randint(255, size=n_values, dtype="uint8"),
            },
            index=index,
        )

        if nlp:
            df["nlp"] = random.sample(example_diseases, k=n_values)

        return df

    obs = _generate_typed_df(M, obs_names, nlp=include_nlp)
    var = _generate_typed_df(N, var_names, nlp=include_nlp)

    obs.rename(columns={"cat": "obs_cat"}, inplace=True)
    var.rename(columns={"cat": "var_cat"}, inplace=True)

    if X_type is None:
        X = None
    else:
        if include_nlp:
            X_np_array = np.random.binomial(100, 0.005, (M, N - 1)).astype(object)
            X = np.append(X_np_array, [[el] for el in random.sample(example_diseases, k=M)], axis=1)
        else:
            X_np_array = np.random.binomial(100, 0.005, (M, N))
            X = X_type(X_np_array).astype(X_dtype)

    obsm = {
        "array": np.random.random((M, 50)),
        "sparse": sparse.random(M, 100, format="csr"),
        "df": _generate_typed_df(M, obs_names),
    }
    obsm = {k: v for k, v in obsm.items() if type(v) in obsm_types}
    varm = {
        "array": np.random.random((N, 50)),
        "sparse": sparse.random(N, 100, format="csr"),
        "df": _generate_typed_df(N, var_names),
    }
    varm = {k: v for k, v in varm.items() if type(v) in varm_types}
    layers = {"array": np.random.random((M, N)), "sparse": sparse.random(M, N, format="csr")}
    layers = {k: v for k, v in layers.items() if type(v) in layers_types}
    obsp = {"array": np.random.random((M, M)), "sparse": sparse.random(M, M, format="csr")}
    varp = {"array": np.random.random((N, N)), "sparse": sparse.random(N, N, format="csr")}

    def _generate_vstr_recarray(m, n, dtype=None):
        size = m * n
        lengths = np.random.randint(3, 5, size)
        letters = np.array(list(ascii_letters))
        gen_word = lambda w: "".join(np.random.choice(letters, w))
        arr = np.array([gen_word(length) for length in lengths]).reshape(m, n)

        return pd.DataFrame(arr, columns=[gen_word(5) for _ in range(n)]).to_records(index=False, column_dtypes=dtype)

    uns = {
        "O_recarray": _generate_vstr_recarray(N, 5),
        "nested": {
            "scalar_str": "str",
            "scalar_int": 42,
            "scalar_float": 3.0,
            "nested_further": {"array": np.arange(5)},
        },
    }

    if include_nlp:
        X_dtype = np.dtype(object)

    adata = AnnData(
        X=X,
        obs=obs,
        var=var,
        obsm=obsm,
        varm=varm,
        layers=layers,
        obsp=obsp,
        varp=varp,
        uns=uns,
    )

    logg.info(f"Generated an AnnData object with n_obs x n_vars = `{adata.n_obs}` x `{adata.n_vars}`.")

    return adata


def get_obs_df(  # pragma: no cover
    adata: AnnData,
    keys: Iterable[str] = (),
    obsm_keys: Iterable[tuple[str, int]] = (),
    *,
    layer: str = None,
    features: str = None,
):
    """Return values for observations in adata.

    Args:
        adata: AnnData object to get values from.
        keys: Keys from either `.var_names`, `.var[gene_symbols]`, or `.obs.columns`.
        obsm_keys: Tuple of `(key from obsm, column index of obsm[key])`.
        layer: Layer of `adata`.
        features: Column of `adata.var` to search for `keys` in.

    Returns:
        A dataframe with `adata.obs_names` as index, and values specified by `keys` and `obsm_keys`.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ages = ep.ad.get_obs_df(adata, keys=["age"])
    """
    return obs_df(adata=adata, keys=keys, obsm_keys=obsm_keys, layer=layer, gene_symbols=features)


def get_var_df(  # pragma: no cover
    adata: AnnData,
    keys: Iterable[str] = (),
    varm_keys: Iterable[tuple[str, int]] = (),
    *,
    layer: str = None,
):
    """Return values for observations in adata.

    Args:
        adata: AnnData object to get values from.
        keys: Keys from either `.obs_names`, or `.var.columns`.
        varm_keys: Tuple of `(key from varm, column index of varm[key])`.
        layer: Layer of `adata`.

    Returns:
        A dataframe with `adata.var_names` as index, and values specified by `keys` and `varm_keys`.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> four_patients = ep.ad.get_var_df(adata, keys=["0", "1", "2", "3"])
    """
    return var_df(adata=adata, keys=keys, varm_keys=varm_keys, layer=layer)


def get_rank_features_df(
    adata: AnnData,
    group: str | Iterable[str],
    *,
    key: str = "rank_features_groups",
    pval_cutoff: float | None = None,
    log2fc_min: float | None = None,
    log2fc_max: float | None = None,
    features: str | None = None,
):
    """:func:`ehrapy.tl.rank_features_groups` results in the form of a :class:`~pandas.DataFrame`.

    Args:
        adata: AnnData object to get values from.
        group: Which group (as in :func:`ehrapy.tl.rank_genes_groups`'s `groupby` argument)
               to return results from. Can be a list. All groups are returned if groups is `None`.
        key: Key differential groups were stored under.
        pval_cutoff: Return only adjusted p-values below the  cutoff.
        log2fc_min: Minimum logfc to return.
        log2fc_max: Maximum logfc to return.
        features: Column name in `.var` DataFrame that stores gene symbols.
                  Specifying this will add that column to the returned DataFrame.

    Returns:
        A Pandas DataFrame of all rank genes groups results.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.tl.rank_features_groups(adata, "service_unit")
        >>> df = ep.ad.get_rank_features_df(adata, group="FICU")
    """
    return rank_genes_groups_df(
        adata=adata,
        group=group,
        key=key,
        pval_cutoff=pval_cutoff,
        log2fc_min=log2fc_min,
        log2fc_max=log2fc_max,
        gene_symbols=features,
    )


class NotEncodedError(AssertionError):
    pass
