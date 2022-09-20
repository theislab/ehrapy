from __future__ import annotations

import random
from collections import OrderedDict
from string import ascii_letters
from typing import Collection, NamedTuple

import numpy as np
import pandas as pd
from anndata import AnnData, concat
from mudata import MuData
from rich import print
from rich.text import Text
from rich.tree import Tree
from scipy import sparse
from scipy.sparse import issparse
from ehrapy import logger as logg


class BaseDataframes(NamedTuple):
    obs: pd.DataFrame
    df: pd.DataFrame


def df_to_anndata(
    df: pd.DataFrame, columns_obs_only: list[str] | None = None, index_column: str | None = None
) -> AnnData:
    """Transform a given pandas dataframe into an AnnData object. Note that columns containing boolean values (either 0/1 or T(t)rue/F(f)alse)
    will be stored as boolean columns whereas the other non numerical columns will be stored as categorical values.

    Args:
        df: The pandas dataframe to be transformed
        columns_obs_only: An optional list of column names that should belong to obs only and not X
        index_column: The index column of obs. This can be either a column name (or its numerical index in the dataframe) or the index of the dataframe

    Returns:
        An AnnData object created from the given pandas dataframe
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
                raise IndexNotFoundError(f"Did not find column {index_column} in neither index or columns!")
        # index_column is neither in the index or in the columns or passed as some value that could not be understood
        else:
            raise IndexNotFoundError(f"Did not find column {index_column} in neither index or columns!")

    # move columns from the input dataframe to later obs
    dataframes = _move_columns_to_obs(df, columns_obs_only)
    numerical_columns = list(dataframes.df.select_dtypes("number").columns)
    # if data is numerical only, short-circuit AnnData creation to have float dtype instead of object
    all_num = True if len(numerical_columns) == len(list(dataframes.df.columns)) else False
    X = dataframes.df.to_numpy(copy=True)

    # initializing an OrderedDict with a non-empty dict might not be intended,
    # see: https://stackoverflow.com/questions/25480089/right-way-to-initialize-an-ordereddict-using-its-constructor-such-that-it-retain/25480206
    uns = OrderedDict()
    # store all numerical/non-numerical columns that are not obs only
    binary_columns = _detect_binary_columns(df, numerical_columns)
    uns["numerical_columns"] = list(set(numerical_columns) ^ set(binary_columns))
    uns["non_numerical_columns"] = list(set(dataframes.df.columns) ^ set(uns["numerical_columns"]))

    # cast non numerical obs only columns to category or bool dtype, which is needed for writing to .h5ad files
    adata = AnnData(
        X=X,
        obs=_cast_obs_columns(dataframes.obs),
        var=pd.DataFrame(index=list(dataframes.df.columns)),
        dtype="float32" if all_num else "object",
        layers={"original": X.copy()},
        uns=uns,
    )

    logg.info(
        f"Transformed given dataframe into an AnnData object with n_obs x n_vars = `{adata.n_obs}` x `{adata.n_vars}`"
    )

    return adata


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
            logg.info(
                f"Columns `{columns_obs_only}` were successfully moved to `obs`"
            )
        except KeyError:
            raise ColumnNotFoundError from KeyError(
                "One or more column names passed to column_obs_only were not found in the input data. "
                "Are the column names spelled correctly?"
            )
    else:
        obs = pd.DataFrame(index=df.index.map(str))
        logg.info(
            f"All columns were successfully moved to `obs`"
        )

    return BaseDataframes(obs, df)


def anndata_to_df(
    adata: AnnData, layer: str = None, obs_cols: list[str] | str | None = None, var_cols: list[str] | str | None = None
) -> pd.DataFrame:
    """Transform an AnnData object to a pandas dataframe.

    Args:
        adata: The AnnData object to be transformed into a pandas Dataframe
        layer: The layer to use for X.
        obs_cols: List of obs columns to add to X.
        var_cols: List of var columns to add to X.

    Returns:
        The AnnData object as a pandas Dataframe
    """
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X
    if issparse(X):
        X = X.toarray()

    df = pd.DataFrame(X, columns=list(adata.var_names))
    if obs_cols:
        if len(adata.obs.columns) == 0:
            raise ObsEmptyError("Cannot slice columns from empty obs!")
        if isinstance(obs_cols, str):
            obs_cols = list(obs_cols)
        if isinstance(obs_cols, list):
            obs_slice = adata.obs[obs_cols]
        # reset index needed since we slice all or at least some columns from obs DataFrame
        obs_slice = obs_slice.reset_index(drop=True)
        df = pd.concat([df, obs_slice], axis=1)

        logg.info(
            f"Added `{obs_cols}` columns to `X`."
        )
    if var_cols:
        if len(adata.var.columns) == 0:
            raise VarEmptyError("Cannot slice columns from empty var!")
        if isinstance(var_cols, str):
            var_cols = list(var_cols)
        if isinstance(var_cols, list):
            var_slice = adata.var[var_cols]
        # reset index needed since we slice all or at least some columns from var DataFrame
        var_slice = var_slice.reset_index(drop=True)
        df = pd.concat([df, var_slice], axis=1)

        logg.info(
            f"Added `{var_cols}` columns to `X`."
        )

    logg.info(
        f"AnnData object was transformed to a pandas dataframe."
    )

    return df


def move_to_obs(adata: AnnData, to_obs: list[str] | str, copy: bool = False) -> AnnData:
    """Move features from X to obs inplace. Note that columns containing boolean values (either 0/1 or True(true)/False(false))
    will be stored as boolean columns whereas the other non numerical columns will be stored as category.
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
    # cast numerical values from object
    adata.obs[num_var] = adata.obs[num_var].apply(pd.to_numeric, errors="ignore", downcast="float")
    # cast non numerical values from object to either bool (if possible) or category
    adata.obs = _cast_obs_columns(adata.obs)
    adata.uns["numerical_columns"] = updated_num_uns
    adata.uns["non_numerical_columns"] = updated_non_num_uns

    logg.info(
        f"Moved `{to_obs}` columns to `obs`."
    )

    if copy:
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

    logg.info(
        f"Moved `{to_x}` features to `X`."
    )

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


def type_overview(
    data: MuData | AnnData, sort_by: str | None = None, sort_reversed: bool = False
) -> None:  # pragma: no cover
    """Prints the current state of an :class:`~anndata.AnnData` or :class:`~mudata.MuData` object in a tree format.
    Output could be printed in sorted format by using one of `dtype`, `order`, `num_cats` or `None`, which sorts by data type, lexicographical order,
    number of unique values (excluding NaN's) and unsorted respectively. Note that sorting by `num_cats` only affects
    encoded variables currently and will display unencoded vars unsorted.

    Args:
        data: :class:`~anndata.AnnData` or :class:`~mudata.MuData` object to display
        sort_by: How the tree output should be sorted. One of `dtype`, `order`, `num_cats` or None (defaults to None -> unsorted)
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
        raise EhrapyRepresentationError(
            f"Unable to present object of type {type(data)}. Can only display AnnData or MuData objects!"
        )


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
            sorted_by_type = {
                var: _type
                for var, _type in sorted(
                    dtype_dict.items(), key=lambda item: item[1][0 if sort_by == "dtype" else 1], reverse=sort_reversed
                )
            }
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
        var_names = sorted(list(adata.var_names.values), reverse=sort_reversed)
        _sort_by_order_or_none(adata, branch_num, var_names)
    elif sort_by == "dtype":
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


def _mudata_type_overview(
    mudata: MuData, sort: str | None = None, sort_reversed: bool = False
) -> None:  # pragma: no cover
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
    """Add branches to tree for sorting by order or unsorted."""
    var_names_val = list(adata.var_names.values)
    for other_vars in var_names:
        if not other_vars.startswith("ehrapycat"):
            idx = var_names_val.index(other_vars)
            unique_categoricals = pd.unique(adata.X[:, idx: idx + 1].flatten())
            data_type = pd.api.types.infer_dtype(unique_categoricals)
            branch.add(f"[blue]{other_vars} -> [green]data type: [blue]{data_type}")


def _sort_by_type(adata: AnnData, branch, var_names: list[str], sort_reversed: bool):
    """Sort tree output by datatype"""
    tmp_dict = {}
    var_names_val = list(adata.var_names.values)

    for other_vars in var_names:
        if not other_vars.startswith("ehrapycat"):
            idx = var_names_val.index(other_vars)
            unique_categoricals = pd.unique(adata.X[:, idx: idx + 1].flatten())
            data_type = pd.api.types.infer_dtype(unique_categoricals)
            tmp_dict[other_vars] = data_type

    sorted_by_type = {
        var: _type for var, _type in sorted(tmp_dict.items(), key=lambda item: item[1], reverse=sort_reversed)
    }
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
    _assert_encoded(adata)

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

    logg.info(
        f"Column names for numeric variables {vars} were replaced."
    )

    return adata


def _update_uns(
    adata: AnnData, moved_columns: list[str], to_x: bool = False
) -> tuple[list[str], list[str], list[str] | None]:
    """Update .uns of adata to reflect the changes made on the object by moving columns from X to obs or vice versa.

    1.) Moving `col1` from `X` to `obs`: `col1` is either numerical or non_numerical, so delete it from the corresponding entry in `uns`
    2.) Moving `col1` from `obs` to `X`: `col1` is either numerical or non_numerical, so add it to the corresponding entry in `uns`
    """
    moved_columns_set = set(moved_columns)
    num_set = set(adata.uns["numerical_columns"].copy())
    non_num_set = set(adata.uns["non_numerical_columns"].copy())
    if not to_x:
        var_num = []
        for var in moved_columns:
            if var in num_set:
                var_num.append(var)
                num_set -= {var}
            elif var in non_num_set:
                non_num_set -= {var}
        logg.info(
            f"Moved `{moved_columns}` columns to `X`."
        )
        return list(num_set), list(non_num_set), var_num
    else:
        all_moved_non_num_columns = moved_columns_set ^ set(adata.obs.select_dtypes("number").columns)
        all_moved_num_columns = list(moved_columns_set ^ all_moved_non_num_columns)
        logg.info(
            f"Moved `{moved_columns}` columns to `obs`."
        )
        return all_moved_num_columns, list(all_moved_non_num_columns), None


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


def generate_anndata(
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

    Example:
        .. code-block:: python

            import ehrapy as ep

            adata = ep.ad.generate_anndata((2, 2), include_nlp=True)
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
            dict(
                cat=pd.Categorical(np.random.choice(letters, n_values)),
                cat_ordered=pd.Categorical(np.random.choice(letters, n_values), ordered=True),
                int64=np.random.randint(-50, 50, n_values),
                float64=np.random.random(n_values),
                uint8=np.random.randint(255, size=n_values, dtype="uint8"),
            ),
            index=index,
        )

        if nlp:
            df["nlp"] = random.sample(example_diseases, k=n_values)

        return df

    obs = _generate_typed_df(M, obs_names, nlp=include_nlp)
    var = _generate_typed_df(N, var_names, nlp=include_nlp)

    obs.rename(columns=dict(cat="obs_cat"), inplace=True)
    var.rename(columns=dict(cat="var_cat"), inplace=True)

    if X_type is None:
        X = None
    else:
        if include_nlp:
            X_np_array = np.random.binomial(100, 0.005, (M, N - 1)).astype(object)
            X = np.append(X_np_array, list(map(lambda el: [el], random.sample(example_diseases, k=M))), axis=1)
        else:
            X_np_array = np.random.binomial(100, 0.005, (M, N))
            X = X_type(X_np_array).astype(X_dtype)

    obsm = dict(
        array=np.random.random((M, 50)),
        sparse=sparse.random(M, 100, format="csr"),
        df=_generate_typed_df(M, obs_names),
    )
    obsm = {k: v for k, v in obsm.items() if type(v) in obsm_types}
    varm = dict(
        array=np.random.random((N, 50)),
        sparse=sparse.random(N, 100, format="csr"),
        df=_generate_typed_df(N, var_names),
    )
    varm = {k: v for k, v in varm.items() if type(v) in varm_types}
    layers = dict(array=np.random.random((M, N)), sparse=sparse.random(M, N, format="csr"))
    layers = {k: v for k, v in layers.items() if type(v) in layers_types}
    obsp = dict(array=np.random.random((M, M)), sparse=sparse.random(M, M, format="csr"))
    varp = dict(array=np.random.random((N, N)), sparse=sparse.random(N, N, format="csr"))

    def _generate_vstr_recarray(m, n, dtype=None):
        size = m * n
        lengths = np.random.randint(3, 5, size)
        letters = np.array(list(ascii_letters))
        gen_word = lambda l: "".join(np.random.choice(letters, l))
        arr = np.array([gen_word(length) for length in lengths]).reshape(m, n)

        return pd.DataFrame(arr, columns=[gen_word(5) for _ in range(n)]).to_records(index=False, column_dtypes=dtype)

    uns = dict(
        O_recarray=_generate_vstr_recarray(N, 5),
        nested=dict(
            scalar_str="str",
            scalar_int=42,
            scalar_float=3.0,
            nested_further=dict(array=np.arange(5)),
        ),
    )

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
        dtype=X_dtype,
        uns=uns,
    )
    logg.info(
        f"Generated an AnnData object with n_obs x n_vars = `{adata.n_obs}` x `{adata.n_vars}`"
    )
    return adata


class NotEncodedError(AssertionError):
    pass


class ColumnNotFoundError(Exception):
    pass


class IndexNotFoundError(Exception):
    pass


class ObsEmptyError(Exception):
    pass


class VarEmptyError(Exception):
    pass


class ObsMoveError(Exception):
    pass


class EhrapyRepresentationError(ValueError):
    pass
