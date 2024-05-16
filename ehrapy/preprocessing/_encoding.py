from __future__ import annotations

from collections import OrderedDict
from itertools import chain
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from lamin_utils import logger
from rich.progress import BarColumn, Progress
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ehrapy.anndata import anndata_to_df, check_feature_types
from ehrapy.anndata._constants import (
    CATEGORICAL_TAG,
    FEATURE_TYPE_KEY,
    NUMERIC_TAG,
)
from ehrapy.anndata.anndata_ext import _get_var_indices_for_type

available_encodings = {"one-hot", "label"}


@check_feature_types
def encode(
    adata: AnnData,
    autodetect: bool | dict = False,
    encodings: dict[str, dict[str, list[str]]] | dict[str, list[str]] | str | None = "one-hot",
) -> AnnData:
    """Encode categoricals of an :class:`~anndata.AnnData` object.

    Categorical values could be either passed via parameters or are autodetected on the fly.
    The categorical values are also stored in obs and uns (for keeping the original, unencoded values).
    The current encoding modes for each variable are also stored in adata.var['encoding_mode'].
    Variable names in var are updated according to the encoding modes used. A variable name starting with `ehrapycat_`
    indicates an encoded column (or part of it).

    Autodetect mode:
        By using this mode, every column that contains non-numerical values is encoded.
        In addition, every binary column will be encoded too.
        These are those columns which contain only 1's and 0's (could be either integers or floats).

    Available encodings are:
        1. one-hot (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
        2. label (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

    Args:
        adata: A :class:`~anndata.AnnData` object.
        autodetect: Whether to autodetect categorical values that will be encoded.
        encodings: Only needed if autodetect set to False.
                   A dict containing the encoding mode and categorical name for the respective column
                   or the specified encoding that will be applied to all columns.

    Returns:
        An :class:`~anndata.AnnData` object with the encoded values in X.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2()
        >>> adata_encoded = ep.pp.encode(adata, autodetect=True, encodings="one-hot")

        >>> # Example using custom encodings per columns:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2()
        >>> # encode col1 and col2 using label encoding and encode col3 using one hot encoding
        >>> adata_encoded = ep.pp.encode(
        ...     adata, autodetect=False, encodings={"label": ["col1", "col2"], "one-hot": ["col3"]}
        ... )
    """
    if isinstance(adata, AnnData):
        if isinstance(encodings, str) and not autodetect:
            raise ValueError("Passing a string for parameter encodings is only possible when using autodetect=True!")
        elif autodetect and not isinstance(encodings, (str, type(None))):
            raise ValueError(
                f"Setting encode mode with autodetect=True only works by passing a string (encode mode name) or None not {type(encodings)}!"
            )
        if "original" not in adata.layers.keys():
            adata.layers["original"] = adata.X.copy()

        # autodetect categorical values based on feature types stored in adata.var[FEATURE_TYPE_KEY]
        if autodetect:
            categoricals_names = _get_var_indices_for_type(adata, CATEGORICAL_TAG)

            if "encoding_mode" in adata.var.keys():
                if adata.var["encoding_mode"].isnull().values.any():
                    not_encoded_features = adata.var["encoding_mode"].isna().index
                    categoricals_names = [cat for cat in categoricals_names if cat in not_encoded_features]
                else:
                    logger.warning(
                        "The current AnnData object has been already encoded. Returning original AnnData object!"
                    )
                    return adata

            # filter out categorical columns, that are already stored numerically
            df_adata = anndata_to_df(adata)
            categoricals_names = [
                feat
                for feat in categoricals_names
                if not np.all(df_adata[feat].apply(type).isin([int, float, complex]))
            ]

            # no columns were detected, that would require an encoding (e.g. non-numerical columns)
            if not categoricals_names:
                logger.warning("Detected no columns that need to be encoded. Leaving passed AnnData object unchanged.")
                return adata
            # copy uns so it can be used in encoding process without mutating the original anndata object
            updated_obs = _update_obs(adata, categoricals_names)

            encoded_x = None
            encoded_var_names = adata.var_names.to_list()
            unencoded_var_names = adata.var_names.to_list()
            if encodings not in available_encodings:
                raise ValueError(
                    f"Unknown encoding mode {encodings}. Please provide one of the following encoding modes:\n"
                    f"{available_encodings}"
                )
            single_encode_mode_switcher = {
                "one-hot": _one_hot_encoding,
                "label": _label_encoding,
            }
            with Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                refresh_per_second=1500,
            ) as progress:
                task = progress.add_task(f"[red]Running {encodings} on detected columns ...", total=1)
                # encode using the desired mode
                encoded_x, encoded_var_names, unencoded_var_names = single_encode_mode_switcher[encodings](  # type: ignore
                    adata,
                    encoded_x,
                    updated_obs,
                    encoded_var_names,
                    unencoded_var_names,
                    categoricals_names,
                    progress,
                    task,
                )
                progress.update(task, description="Updating layer originals ...")

                # update layer content with the latest categorical encoding and the old other values
                updated_layer = _update_layer_after_encoding(
                    adata.layers["original"],
                    encoded_x,
                    encoded_var_names,
                    adata.var_names.to_list(),
                    categoricals_names,
                )
                progress.update(task, description=f"[bold blue]Finished {encodings} of autodetected columns.")

                # copy non-encoded columns, and add new tag for encoded columns. This is needed to track encodings
                new_var = pd.DataFrame(index=encoded_var_names)
                new_var[FEATURE_TYPE_KEY] = adata.var[FEATURE_TYPE_KEY].copy()
                new_var[FEATURE_TYPE_KEY].loc[new_var.index.str.contains("ehrapycat")] = CATEGORICAL_TAG

                # store unencoded columns in var
                new_var["unencoded_var_names"] = unencoded_var_names

                # store encoding mode in var
                new_var["encoding_mode"] = [
                    encodings if var in categoricals_names else None for var in unencoded_var_names
                ]

                encoded_ann_data = AnnData(
                    encoded_x,
                    obs=updated_obs,
                    var=new_var,
                    uns=adata.uns.copy(),
                    layers={"original": updated_layer},
                )

        # user passed categorical values with encoding mode for each of them
        else:
            # re-encode data
            if "encoding_mode" in adata.var.keys():
                encodings = _reorder_encodings(adata, encodings)  # type: ignore
                adata = _undo_encoding(adata)

            # are all specified encodings valid?
            for encoding in encodings.keys():  # type: ignore
                if encoding not in available_encodings:
                    raise ValueError(
                        f"Unknown encoding mode {encoding}. Please provide one of the following encoding modes:\n"
                        f"{available_encodings}"
                    )

            categoricals = list(chain(*encodings.values()))  # type: ignore

            # ensure no categorical column gets encoded twice
            if len(categoricals) != len(set(categoricals)):
                raise ValueError(
                    "The categorical column names given contain at least one duplicate column. "
                    "Check the column names to ensure that no column is encoded twice!"
                )
            elif any(cat in adata.var_names[adata.var[FEATURE_TYPE_KEY] == NUMERIC_TAG] for cat in categoricals):
                logger.warning(
                    "At least one of passed column names seems to have numerical dtype. In general it is not recommended "
                    "to encode numerical columns!"
                )

            updated_obs = _update_obs(adata, categoricals)

            encoding_mode = {}
            encoded_x = None
            encoded_var_names = adata.var_names.to_list()
            unencoded_var_names = adata.var_names.to_list()
            with Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                refresh_per_second=1500,
            ) as progress:
                for encoding in encodings.keys():  # type: ignore
                    task = progress.add_task(f"[red]Setting up {encodings}", total=1)
                    encode_mode_switcher = {
                        "one-hot": _one_hot_encoding,
                        "label": _label_encoding,
                    }
                    progress.update(task, description=f"Running {encoding} ...")
                    # perform the actual encoding
                    encoded_x, encoded_var_names, unencoded_var_names = encode_mode_switcher[encoding](
                        adata,
                        encoded_x,
                        updated_obs,
                        encoded_var_names,
                        unencoded_var_names,
                        encodings[encoding],  # type: ignore
                        progress,
                        task,  # type: ignore
                    )

                    for categorical in encodings[encoding]:  # type: ignore
                        categorical = [categorical] if isinstance(categorical, str) else categorical  # type: ignore
                        for column_name in categorical:
                            # get idx of column in unencoded_var_names
                            indices = [i for i, var in enumerate(unencoded_var_names) if var == column_name]
                            encoded_var = [encoded_var_names[idx] for idx in indices]
                            for var in encoded_var:
                                encoding_mode[var] = encoding

            # update original layer content with the new categorical encoding and the old other values
            updated_layer = _update_layer_after_encoding(
                adata.layers["original"],
                encoded_x,
                encoded_var_names,
                adata.var_names.to_list(),
                categoricals,
            )

            # copy non-encoded columns, and add new tag for encoded columns. This is needed to track encodings
            new_var = pd.DataFrame(index=encoded_var_names)

            new_var[FEATURE_TYPE_KEY] = adata.var[FEATURE_TYPE_KEY].copy()
            new_var[FEATURE_TYPE_KEY].loc[new_var.index.str.contains("ehrapycat")] = CATEGORICAL_TAG

            # store unencoded columns in var
            new_var["unencoded_var_names"] = unencoded_var_names

            # update encoding mode in var, keeping already annotated columns
            if "encoding_mode" in adata.var.keys():
                encoding_mode.update(
                    {key: value for key, value in adata.var["encoding_mode"].items() if value is not None}
                )
            new_var["encoding_mode"] = [None] * len(new_var)
            for categorical in encoding_mode.keys():
                new_var["encoding_mode"].loc[categorical] = encoding_mode[categorical]

            try:
                encoded_ann_data = AnnData(
                    X=encoded_x,
                    obs=updated_obs,
                    var=new_var,
                    uns=adata.uns.copy(),
                    layers={"original": updated_layer},
                )

            # if the user did not pass every non-numerical column for encoding, an Anndata object cannot be created
            except ValueError:
                raise AnnDataCreationError(
                    "Creation of AnnData object failed. Ensure that you passed all non numerical, "
                    "categorical values for encoding!"
                ) from None

        encoded_ann_data.X = encoded_ann_data.X.astype(np.float32)

        return encoded_ann_data
    else:
        raise ValueError(f"Cannot encode object of type {type(adata)}. Can only encode AnnData objects!")


def _one_hot_encoding(
    adata: AnnData,
    X: np.ndarray | None,
    updated_obs: pd.DataFrame,
    var_names: list[str],
    unencoded_var_names: list[str],
    categories: list[str],
    progress: Progress,
    task,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Encode categorical columns using one hot encoding.

    Args:
        adata: The current AnnData object
        X: Current (encoded) X
        updated_obs: A copy of the original obs where the original categorical values are stored that will be encoded
        var_names: Var names of current AnnData object
        categories: The name of the categorical columns to be encoded

    Returns:
        Encoded new X and the corresponding new var names
    """
    original_values = _initial_encoding(updated_obs, categories)
    progress.update(task, description="[bold blue]Running one-hot encoding on passed columns ...")

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(original_values)
    categorical_prefixes = [
        f"ehrapycat_{category}_{str(suffix).strip()}"
        for idx, category in enumerate(categories)
        for suffix in encoder.categories_[idx]
    ]
    unencoded_prefixes = [category for idx, category in enumerate(categories) for suffix in encoder.categories_[idx]]
    transformed = encoder.transform(original_values)
    # X is None if this is the first encoding "round" -> take the former X
    if X is None:
        X = adata.X
    progress.advance(task, 1)
    progress.update(task, description="[blue]Updating X and var ...")

    temp_x, temp_var_names, unencoded_var_names = _update_encoded_data(
        X, transformed, var_names, categorical_prefixes, categories, unencoded_prefixes, unencoded_var_names
    )
    progress.update(task, description="[blue]Finished one-hot encoding.")

    return temp_x, temp_var_names, unencoded_var_names


def _label_encoding(
    adata: AnnData,
    X: np.ndarray | None,
    updated_obs: pd.DataFrame,
    var_names: list[str],
    unencoded_var_names: list[str],
    categoricals: list[str],
    progress: Progress,
    task,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Encode categorical columns using label encoding.

    Args:
        adata: The current AnnData object
        X: Current (encoded) X
        updated_obs: A copy of the original obs where the original categorical values are stored that will be encoded
        var_names: Var names of current AnnData object
        categoricals: The name of the categorical columns, that need to be encoded

    Returns:
        Encoded new X and the corresponding new var names
    """
    original_values = _initial_encoding(updated_obs, categoricals)
    # label encoding expects input array to be 1D, so iterate over all columns and encode them one by one
    for idx in range(original_values.shape[1]):
        progress.update(task, description=f"[blue]Running label encoding on column {categoricals[idx]} ...")
        label_encoder = LabelEncoder()
        row_vec = original_values[:, idx : idx + 1].ravel()  # type: ignore
        label_encoder.fit(row_vec)
        transformed = label_encoder.transform(row_vec)
        # need a column vector instead of row vector
        original_values[:, idx : idx + 1] = transformed[..., None]
        progress.advance(task, 1 / len(categoricals))
    category_prefixes = [f"ehrapycat_{categorical}" for categorical in categoricals]
    # X is None if this is the first encoding "round" -> take the former X
    if X is None:
        X = adata.X

    progress.update(task, description="[blue]Updating X and var ...")
    temp_x, temp_var_names, unencoded_var_names = _update_encoded_data(
        X, original_values, var_names, category_prefixes, categoricals, categoricals, unencoded_var_names
    )
    progress.update(task, description="[blue]Finished label encoding.")

    return temp_x, temp_var_names, unencoded_var_names


def _update_layer_after_encoding(
    old_layer: np.ndarray,
    new_x: np.ndarray,
    new_var_names: list[str],
    old_var_names: list[str],
    categories: list[str],
) -> np.ndarray:
    """Update the original layer containing the initial non categorical values and the latest encoded categoricals.

    Args:
        old_layer: The previous "original" layer
        new_x: The new encoded X
        new_var_names: The new encoded var names
        old_var_names: The previous var names
        categories: All previous categorical names

    Returns
        A Numpy array containing all numericals together with all encoded categoricals.
    """
    try:
        # get the index of the first column of the new encoded X, that does not store an encoded categorical
        new_cat_stop_index = next(i for i in range(len(new_var_names)) if not new_var_names[i].startswith("ehrapycat"))
        # get the index of the first column of the old encoded X, that does not store an encoded categorical
        old_cat_stop_index = next(i for i in range(len(old_var_names)) if not old_var_names[i].startswith("ehrapycat"))
    # when there are only encoded columns, simply return a copy of the new X, since to originals will be kept in the layer
    except StopIteration:
        return new_x.copy().astype("float32")
    # keep track of all indices with original value columns, that are (and were) not encoded
    idx_list = []
    for idx, col_name in enumerate(old_var_names[old_cat_stop_index:]):
        # this case is needed when there are one or more numerical (but categorical) columns that was not encoded yet
        if col_name not in categories:
            idx_list.append(idx + old_cat_stop_index)
    # slice old original layer using the selector
    old_layer_view = old_layer[:, idx_list]
    # get all encoded categoricals of X
    encoded_categoricals = new_x[:, :new_cat_stop_index]
    # horizontally stack all encoded categoricals and the remaining "old original values"
    updated_layer = np.hstack((encoded_categoricals, old_layer_view))

    try:
        logger.info("Updated the original layer after encoding.")
        return updated_layer.astype("float32")
    except ValueError as e:
        raise ValueError("Ensure that all columns which require encoding are being encoded.") from e


def _update_encoded_data(
    X: np.ndarray,
    transformed: np.ndarray,
    var_names: list[str],
    categorical_prefixes: list[str],
    categoricals: list[str],
    unencoded_prefixes: list[str],
    unencoded_var_names: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Update X and var_names after each encoding.

    Args:
        X: Current (former) X
        transformed: The encoded (transformed) categorical column
        var_names: Var names of current AnnData object
        categorical_prefixes: The name(s) of the encoded column(s)
        categoricals: The categorical values that were encoded recently
        unencoded_prefixes: The unencoded names of the categorical columns that were encoded

    Returns:
        Encoded new X, the corresponding new var names, and the unencoded var names
    """
    idx = _get_categoricals_old_indices(var_names, categoricals)
    # delete the original categorical column
    del_cat_column_x = np.delete(X, list(idx), 1)
    # create the new, encoded X
    temp_x = np.hstack((transformed, del_cat_column_x))
    # delete old categorical name
    var_names = [col_name for col_idx, col_name in enumerate(var_names) if col_idx not in idx]
    temp_var_names = categorical_prefixes + var_names

    unencoded_var_names = [col_name for col_idx, col_name in enumerate(unencoded_var_names) if col_idx not in idx]
    unencoded_var_names = unencoded_prefixes + unencoded_var_names

    return temp_x, temp_var_names, unencoded_var_names


def _initial_encoding(
    obs: pd.DataFrame,
    categoricals: list[str],
) -> np.ndarray:
    """Get all original values for all categoricals that need to be encoded (again).

    Args:
        obs: A copy of the original obs where the original categorical values are stored that will be encoded
        categoricals: All categoricals that need to be encoded

    Returns:
        Numpy array of all original categorial values
    """
    # create numpy array from all original categorical values, that will be encoded (again)
    array = np.array([obs[categoricals[i]].ravel() for i in range(len(categoricals))]).transpose()

    return array


def _undo_encoding(
    adata: AnnData,
    verbose: bool = True,
) -> AnnData | None:
    """Undo the current encodings applied to all columns in X. This currently resets the AnnData object to its initial state.

    Args:
        adata: The AnnData object
        verbose: Set to False to suppress warnings. Defaults to True.

    Returns:
        A (partially) encoding reset AnnData object
    """
    if "encoding_mode" not in adata.var.keys():
        if verbose:
            logger.warning("Calling undo_encoding on unencoded AnnData object. Returning original AnnData object!")
        return adata

    # get all encoded features
    categoricals = _get_encoded_features(adata)

    # get all columns that should be stored in obs only
    columns_obs_only = [column_name for column_name in list(adata.obs.columns) if column_name not in categoricals]

    transformed = _initial_encoding(adata.obs, categoricals)
    temp_x, temp_var_names = _delete_all_encodings(adata)
    new_x = np.hstack((transformed, temp_x)) if temp_x is not None else transformed
    new_var_names = categoricals + temp_var_names if temp_var_names is not None else categoricals

    # only keep columns in obs that were stored in obs only -> delete every encoded column from obs
    new_obs = adata.obs[columns_obs_only]

    var = pd.DataFrame(index=new_var_names)
    var[FEATURE_TYPE_KEY] = [
        adata.var.loc[adata.var["unencoded_var_names"] == unenc_var_name, FEATURE_TYPE_KEY].unique()[0]
        for unenc_var_name in new_var_names
    ]

    return AnnData(
        new_x,
        obs=new_obs,
        var=var,
        uns=OrderedDict(),
        layers={"original": new_x.copy()},
    )


def _delete_all_encodings(adata: AnnData) -> tuple[np.ndarray | None, list | None]:
    """Delete all encoded columns and keep track of their indices.

    Args:
        adata: The AnnData object to operate on

    Returns:
        A temporary X were all encoded columns are deleted and all var_names of unencoded columns.
    """
    var_names = list(adata.var_names)
    if adata.X is not None and var_names is not None:
        idx = 0
        for var in var_names:
            if not var.startswith("ehrapycat"):
                break
            idx += 1
        # case: only encoded columns were found
        if idx == len(var_names):
            return None, None
        # don't need to consider case when no encoded columns are there, since undo_encoding would not run anyways in this case
        return adata.X[:, idx:].copy(), var_names[idx:]
    return None, None


def _reorder_encodings(adata: AnnData, new_encodings: dict[str, list[list[str]] | list[str]]):
    """Reorder the encodings and update which column will be encoded using which mode.

    Args:
        adata: The AnnData object to be reencoded
        new_encodings: The new encodings passed by the user (might affect encoded as well as previously non encoded columns)

    Returns:
        An updated encoding scheme
    """
    latest_encoded_columns = sum(new_encodings.values(), [])

    # check for duplicates and raise an error if any
    if len(set(latest_encoded_columns)) != len(latest_encoded_columns):
        logger.error(
            "Reencoding AnnData object failed. You have at least one duplicate in your encodings. A column "
            "cannot be encoded at the same time using different encoding modes!"
        )
        raise DuplicateColumnEncodingError

    encodings = {}
    for encode_mode in available_encodings:
        encoded_categoricals_with_mode = (
            adata.var.loc[adata.var["encoding_mode"] == encode_mode, "unencoded_var_names"].unique().tolist()
        )

        encodings[encode_mode] = new_encodings[encode_mode] if encode_mode in new_encodings.keys() else []
        # add all columns that were encoded with the current mode before and are not reencoded
        encodings[encode_mode] += [cat for cat in encoded_categoricals_with_mode if cat not in latest_encoded_columns]

        if len(encodings[encode_mode]) == 0:
            del encodings[encode_mode]

    return encodings


def _get_categoricals_old_indices(old_var_names: list[str], encoded_categories: list[str]) -> set[int]:
    """Get indices of every (possibly encoded) categorical column belonging to a newly encoded categorical value.

    Args:
        old_var_names: Former variables names
        encoded_categories: Already encoded categories

    Returns:
        Set of all indices of formerly encoded categories belonging to a newly encoded categorical value.
    """
    idx_list = set()
    category_set = set(encoded_categories)
    for idx, old_var_name in enumerate(old_var_names):
        # if the old variable was previously unencoded (only the case for numerical categoricals)
        if old_var_name in category_set:
            idx_list.add(idx)
        # if the old variable was already encoded
        elif old_var_name.startswith("ehrapycat_"):
            if any(old_var_name[10:].startswith(category) for category in category_set):
                idx_list.add(idx)

    return idx_list


def _update_obs(adata: AnnData, categorical_names: list[str]) -> pd.DataFrame:
    """Add the original categorical values to obs.

    Args:
        adata: The original AnnData object
        categorical_names: Name of each categorical column

    Returns:
        Updated obs with the original categorical values added
    """
    updated_obs = adata.obs.copy()
    for idx, var_name in enumerate(adata.var_names):
        if var_name in updated_obs.columns:
            continue
        elif var_name in categorical_names:
            updated_obs[var_name] = adata.X[::, idx : idx + 1].flatten()
            # note: this will count binary columns (0 and 1 only) as well
            # needed for writing to .h5ad files
            if set(pd.unique(updated_obs[var_name])).issubset({False, True, np.NaN}):
                updated_obs[var_name] = updated_obs[var_name].astype("bool")
    # get all non bool object columns and cast them to category dtype
    object_columns = list(updated_obs.select_dtypes(include="object").columns)
    updated_obs[object_columns] = updated_obs[object_columns].astype("category")
    logger.info(f"The original categorical values `{categorical_names}` were added to obs.")

    return updated_obs


def _get_encoded_features(adata: AnnData) -> list[str]:
    """Get all encoded features in an AnnData object.

    Args:
        adata: The AnnData object

    Returns:
        List of all unencoded names of features that were encoded
    """
    encoded_features = [
        unencoded_feature
        for enc_mode, unencoded_feature in adata.var[["encoding_mode", "unencoded_var_names"]].values
        if enc_mode is not None and not pd.isna(enc_mode)
    ]
    return list(set(encoded_features))


class AnnDataCreationError(ValueError):
    pass


class DuplicateColumnEncodingError(ValueError):
    pass
