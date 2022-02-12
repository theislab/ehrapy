from __future__ import annotations

from collections import OrderedDict
from typing import NamedTuple

import numpy as np
import pandas as pd
from anndata import AnnData, concat
from mudata import MuData
from rich import print
from rich.text import Text
from rich.tree import Tree

multi_encoding_modes = {"hash_encoding"}
available_encodings = {"one_hot_encoding", "label_encoding", "count_encoding", *multi_encoding_modes}


class BaseDataframes(NamedTuple):
    obs: pd.DataFrame
    df: pd.DataFrame


def df_to_anndata(
    df: pd.DataFrame, columns_obs_only: list[str] | None = None, index_column: str | None = None
) -> AnnData:
    """Transform a given pandas dataframe into an AnnData object.

    Args:
        df: The pandas dataframe to be transformed
        columns_obs_only: An optional list of column names that should belong to obs only and not X
        index_column: The (optional) index column of obs

    Returns:
        An AnnData object created from the given pandas dataframe
    """
    if index_column:
        df = df.set_index(index_column)
    # move columns from the input dataframe to later obs
    dataframes = _move_columns_to_obs(df, columns_obs_only)
    # if data is numerical only, short-circuit AnnData creation to have float dtype instead of object
    all_num = all(np.issubdtype(column_dtype, np.number) for column_dtype in dataframes.df.dtypes)
    X = dataframes.df.to_numpy(copy=True)
    # initializing an OrderedDict with a non-empty dict might not be intended,
    # see: https://stackoverflow.com/questions/25480089/right-way-to-initialize-an-ordereddict-using-its-constructor-such-that-it-retain/25480206
    uns = OrderedDict()
    # store all numerical/non-numerical columns that are not obs only
    uns["numerical_columns"] = list(dataframes.df.select_dtypes("number").columns)
    uns["non_numerical_columns"] = list(set(dataframes.df.columns) ^ set(uns["numerical_columns"]))
    return AnnData(
        X=X,
        obs=dataframes.obs,
        var=pd.DataFrame(index=list(dataframes.df.columns)),
        dtype="float32" if all_num else "object",
        layers={"original": X.copy()},
        uns=uns,
    )


def _move_columns_to_obs(df: pd.DataFrame, columns_obs_only: list[str] | None) -> BaseDataframes:
    """Move the given columns from the original dataframe (and therefore X) to obs.
    By moving these values will not get lost and will be stored in obs, but will not appear in X.
    This may be useful for textual values like free text.

    Args:
        df: Pandas Dataframe to move the columns for
        columns_obs_only: Columns to move to obs only

    Returns:
        A modified :class:`~pd.DataFrame` object
    """
    if columns_obs_only:
        try:
            obs = df[columns_obs_only].copy()
            obs = obs.set_index(df.index.map(str))
            df = df.drop(columns_obs_only, axis=1)
        except KeyError:
            raise ColumnNotFoundError from KeyError(
                "One or more column names passed to column_obs_only were not found in the input data. "
                "Make sure you spelled the column names correctly."
            )
    else:
        obs = pd.DataFrame(index=df.index.map(str))

    return BaseDataframes(obs, df)


def anndata_to_df(adata: AnnData, add_from_obs: list[str] | str | None = None) -> pd.DataFrame:
    """Transform an AnnData object to a pandas dataframe.

    Args:
        adata: The AnnData object to be transformed into a pandas Dataframe
        add_from_obs: Either "all" or a list of obs names or None, if no columns should be kept from obs

    Returns:
        The AnnData object as a pandas Dataframe
    """
    df = pd.DataFrame(adata.X, columns=list(adata.var_names))
    if add_from_obs:
        if len(adata.obs.columns) == 0:
            raise ObsEmptyError("Cannot slice columns from empty obs!")
        if isinstance(add_from_obs, list):
            obs_slice = adata.obs[add_from_obs]
        else:
            obs_slice = adata.obs
        # reset index needed since we slice all or at least some columns from obs DataFrame
        obs_slice = obs_slice.reset_index(drop=True)
        df = pd.concat([df, obs_slice], axis=1)

    return df


def move_to_obs(adata: AnnData, to_obs: list[str] | str, copy: bool = False) -> AnnData:
    """Move features from X to obs inplace.

    Args:
        adata: The AnnData object
        to_obs: The columns to move to obs
        copy: Whether to return a copy or not

    Returns:
        The original AnnData object with moved columns from X to obs
    """
    if copy:
        adata = adata.copy()

    if isinstance(to_obs, str):  # pragma: no cover
        to_obs = [to_obs]
    # don't allow moving encoded columns as this could lead to inconsistent data in X and obs
    if any(column.startswith("ehrapycat") for column in to_obs):
        raise ObsMoveError(
            "Cannot move encoded columns from X to obs. Either undo encoding or remove them from the list!"
        )
    indices = adata.var_names.isin(to_obs)
    df = adata[:, indices].to_df()
    adata._inplace_subset_var(~indices)
    adata.obs = adata.obs.join(df)
    updated_num_uns, updated_non_num_uns, num_var = _update_uns(adata, to_obs)
    adata.obs[num_var] = adata.obs[num_var].apply(pd.to_numeric, errors="ignore", downcast="float")
    adata.uns["numerical_columns"] = updated_num_uns
    adata.uns["non_numerical_columns"] = updated_non_num_uns

    return adata


def move_to_x(adata: AnnData, to_x: list[str] | str) -> AnnData:
    """Move features from obs to X inplace.

    Args:
        adata: The AnnData object
        to_x: The columns to move to X

    Returns:
        A new AnnData object with moved columns from obs to X. This should not be used for datetime columns currently.
    """
    if isinstance(to_x, str):  # pragma: no cover
        to_x = [to_x]
    new_adata = concat([adata, AnnData(adata.obs[to_x], dtype="object")], axis=1)
    new_adata.obs = adata.obs[adata.obs.columns[~adata.obs.columns.isin(to_x)]]
    # update uns (copy maybe: could be a costly operation but reduces reference cycles)
    # users might save those as separate AnnData object and this could be unexpected behaviour if we dont copy
    num_columns_moved, non_num_columns_moved, _ = _update_uns(adata, to_x, True)
    new_adata.uns["numerical_columns"] = adata.uns["numerical_columns"] + num_columns_moved
    new_adata.uns["non_numerical_columns"] = adata.uns["non_numerical_columns"] + non_num_columns_moved

    return new_adata


def get_column_indices(adata: AnnData, col_names: str | list[str]) -> list[int]:
    """Fetches the column indices in X for a given list of column names

    Args:
        adata: :class:`~anndata.AnnData` object
        col_names: Column names to extract the indices for

    Returns:
        Set of column indices
    """
    if isinstance(col_names, str):  # pragma: no cover
        col_names = [col_names]

    indices = list()
    for idx, col in enumerate(adata.var_names):
        if col in col_names:
            indices.append(idx)

    return indices


def get_column_values(adata: AnnData, indices: int | list[int]) -> np.ndarray:
    """Fetches the column values for a specific index from X

    Args:
        adata: :class:`~anndata.AnnData` object
        indices: The index to extract the values for

    Returns:
        :class:`~numpy.ndarray` object containing the column values
    """
    return np.take(adata.X, indices, axis=1)


def type_overview(data: MuData | AnnData, sort_by: dict[str, str] | str | None = None, sort_reversed: bool = False) -> None:  # pragma: no cover
    """Prints the current state of an :class:`~anndata.AnnData` or :class:`~mudata.MuData` object in a tree format.

    Args:
        data: :class:`~anndata.AnnData` or :class:`~mudata.MuData` object to display
        sort_by: How the tree output should be sorted. One of `data_type`, `encode_mode`, `order` or `None` (defaults to None -> unsorted)
        sort_reversed: Whether to sort in reversed order or not

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.dt.mimic_2(encode=True)
            ep.anndata_ext.type_overview(adata)
    """
    if isinstance(data, AnnData):
        _adata_type_overview(data, sort_by, sort_reversed)
    elif isinstance(data, MuData):
        _mudata_type_overview(data, sort_by, sort_reversed)
    else:
        print(f"[b red]Unable to present object of type {type(data)}. Can only display AnnData or MuData objects!")
        raise EhrapyRepresentationError


def _adata_type_overview(adata: AnnData, sort_by: bool = False, sort_reversed: bool = False) -> None:  # pragma: no cover
    """Display the :class:`~anndata.AnnData object in its current state (encoded and unencoded variables, obs)

    Args:
        adata: The :class:`~anndata.AnnData object to display
        sort_by: Whether to sort output or not
        sort_reversed: Whether to sort output in reversed order or not
    """
    encoding_mapping = {
        encoding: encoding.replace("encoding", "").replace("_", " ").strip() for encoding in available_encodings
    }

    tree = Tree(
        f"[b green]Variable names for AnnData object with {len(adata.obs_names)} obs and {len(adata.var_names)} vars",
        guide_style="underline2 bright_blue",
    )
    if "var_to_encoding" in adata.uns.keys():
        original_values = adata.uns["original_values_categoricals"]
        branch = tree.add("🔐 Encoded variables", style="b green")
        encoded_list = sorted(original_values.keys(), reverse=sort_reversed) if sort_by else list(original_values.keys())
        for categorical in encoded_list:
            unique_categoricals = pd.unique(original_values[categorical].flatten())
            categorical_type = pd.api.types.infer_dtype(unique_categoricals)
            is_nan = pd.DataFrame(unique_categoricals).isnull().values.any()
            branch.add(
                f"[blue]{categorical} -> {len(unique_categoricals) - 1 if is_nan else len(unique_categoricals)} categories;"
                f" [green]{encoding_mapping[adata.uns['var_to_encoding'][categorical]]} [blue]encoded; [green]original data type: [blue]{categorical_type}"
            )

    branch_num = tree.add(Text("🔓 Unencoded variables"), style="b green")

    if sort_by == 'order':
        var_names = sorted(list(adata.var_names.values), reverse=sort_reversed)
        _sort_by_order_or_none(adata, branch_num, var_names)
    elif sort_by == 'data_type':
        var_names = list(adata.var_names.values)
        _sort_by_type(adata, branch_num, var_names, sort_reversed)
    else:
        var_names = list(adata.var_names.values)
        _sort_by_order_or_none(adata, branch_num, var_names)

    if sort_by:
        print(
            "[b yellow]Displaying AnnData object in sorted mode. "
            "Note that this might not be the exact same order of the variables in X or var are stored!"
        )
    print(tree)


def _mudata_type_overview(mudata: MuData, sort: bool = False, sort_reversed: bool = False) -> None:  # pragma: no cover
    """Display the :class:`~mudata.MuData object in its current state (:class:`~anndata.AnnData objects with obs, shapes)

    Args:
        mudata: The :class:`~mudata.MuData object to display
        sort: Whether to sort output or not
        sort_reversed: Whether to sort output in reversed order or not
    """
    tree = Tree(
        f"[b green]Variable names for AnnData object with {len(mudata.obs_names)} obs, {len(mudata.var_names)} vars and {len(mudata.mod.keys())} modalities\n",
        guide_style="underline2 bright_blue",
    )

    modalities = sorted(list(mudata.mod.keys()), reverse=sort_reversed) if sort else list(mudata.mod.keys())
    for mod in modalities:
        branch = tree.add(
            f"[b green]{mod}: [not b blue]n_vars x n_obs: {mudata.mod[mod].n_vars} x {mudata.mod[mod].n_obs}"
        )
        branch.add(
            f"[blue]obs: [black]{', '.join(f'{_single_quote_string(col_name)}' for col_name in mudata.mod[mod].obs.columns)}"
        )
        branch.add(f"[blue]layers: [black]{', '.join(layer for layer in mudata.mod[mod].layers)}\n")
    print(tree)


def _sort_by_order_or_none(adata: AnnData, branch, var_names: list[str]):
    """Add branches to tree for sorting by order or unsorted.
    """
    var_names_val = list(adata.var_names.values)
    for other_vars in var_names:
        if not other_vars.startswith("ehrapycat"):
            idx = var_names_val.index(other_vars)
            unique_categoricals = pd.unique(adata.X[:, idx : idx + 1].flatten())
            data_type = pd.api.types.infer_dtype(unique_categoricals)
            branch.add(f"[blue]{other_vars} -> [green]data type: [blue]{data_type}")


def _sort_by_type(adata: AnnData, branch, var_names: list[str], sort_reversed: bool):
    """Sort tree output by datatype
    """
    tmp_dict = {}
    var_names_val = list(adata.var_names.values)

    for other_vars in var_names:
        if not other_vars.startswith("ehrapycat"):
            idx = var_names_val.index(other_vars)
            unique_categoricals = pd.unique(adata.X[:, idx : idx + 1].flatten())
            data_type = pd.api.types.infer_dtype(unique_categoricals)
            tmp_dict[other_vars] = data_type

    sorted_by_type = {var: _type for var, _type in sorted(tmp_dict.items(), key=lambda item: item[1], reverse=sort_reversed)}
    for var in sorted_by_type:
        branch.add(f"[blue]{var} -> [green]data type: [blue]{sorted_by_type[var]}")


def _single_quote_string(name: str) -> str:  # pragma: no cover
    """Single quote a string to inject it into f-strings, since backslashes cannot be in double f-strings."""
    return f"'{name}'"


def assert_encoded(adata: AnnData):
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
    assert_encoded(adata)

    return adata.uns["numerical_columns"]


def assert_numeric_vars(adata: AnnData, vars: list[str]):
    num_vars = get_numeric_vars(adata)

    try:
        assert set(vars) <= set(num_vars)
    except AssertionError:
        raise ValueError("Some selected vars are not numeric")


def set_numeric_vars(
    adata: AnnData, values: np.ndarray, vars: list[str] | None = None, copy: bool = False
) -> AnnData | None:
    """Sets the column names for numeric variables in X.

    Args:
        adata: :class:`~anndata.AnnData` object
        values: Matrix containing the replacement values
        vars: List of names of the numeric variables to replace. If `None` they will be detected using :func:`~ehrapy.preprocessing.get_numeric_vars`.
        copy: Whether to return a copy with the normalized data.

    Returns:
        :class:`~anndata.AnnData` object with updated X
    """
    assert_encoded(adata)

    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"values must be numeric (current dtype is {values.dtype})")

    n_values = values.shape[1]

    if n_values != len(vars):
        raise ValueError(f"Number of values ({n_values}) does not match number of vars ({len(vars)})")

    if copy:
        adata = adata.copy()

    vars_idx = get_column_indices(adata, vars)

    for i in range(n_values):
        adata.X[:, vars_idx[i]] = values[:, i]

    return adata


def _update_uns(
    adata: AnnData, moved_columns: list[str], to_x: bool = False
) -> tuple[list[str], list[str], list[str] | None]:
    """Update .uns of adata to reflect the changes made on the object by moving columns from X to obs or vice versa.

    1.) Moving `col1` from `X` to `obs`: `col1` is either numerical or non_numerical, so delete it from the corresponding entry in `uns`
    2.) Moving `col1` from `obs` to `X`: `col1` is either numerical or non_numerical, so add it to the corresponding entry in `uns`
    """
    moved_columns_set = set(moved_columns)
    num_set = set(adata.uns["numerical_columns"])
    non_num_set = set(adata.uns["non_numerical_columns"])
    if not to_x:
        var_num = []
        for var in moved_columns:
            if var in num_set:
                var_num.append(var)
                num_set -= {var}
            elif var in non_num_set:
                non_num_set -= {var}
        return list(num_set), list(non_num_set), var_num
    else:
        all_moved_non_num_columns = moved_columns_set ^ set(adata.obs.select_dtypes("number").columns)
        all_moved_num_columns = list(moved_columns_set ^ all_moved_non_num_columns)
        return all_moved_num_columns, list(all_moved_non_num_columns), None


class NotEncodedError(AssertionError):
    pass


class ColumnNotFoundError(Exception):
    pass


class ObsEmptyError(Exception):
    pass


class ObsMoveError(Exception):
    pass


class EhrapyRepresentationError(ValueError):
    pass
