from __future__ import annotations

from collections import OrderedDict
from itertools import chain

import ehrdata as ed
import numpy as np
import pandas as pd
from anndata import AnnData
from ehrdata import EHRData
from ehrdata._logger import logger
from ehrdata.core.constants import CATEGORICAL_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG
from rich.progress import BarColumn, Progress
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ehrapy._compat import _cast_adata_to_match_data_type, function_2D_only, use_ehrdata
from ehrapy.anndata import _check_feature_types
from ehrapy.anndata.anndata_ext import _get_var_indices_for_type

available_encodings = {"one-hot", "label"}


@_check_feature_types
@use_ehrdata(deprecated_after="1.0.0")
@function_2D_only()
def encode(
    edata: EHRData | AnnData,
    autodetect: bool | dict = False,
    encodings: dict[str, list[str]] | str | None = "one-hot",
    *,
    layer: str | None = None,
) -> EHRData | AnnData:
    """Encode categoricals of a data object.

    Categorical values could be either passed via parameters or are autodetected on the fly.
    The categorical values are also stored in obs and uns (for keeping the original, unencoded values).
    The current encoding modes for each variable are also stored in edata.var['encoding_mode'].
    Variable names in var are updated according to the encoding modes used.
    A variable name starting with `ehrapycat_` indicates an encoded column (or part of it).

    Autodetect mode:
        By using this mode, every column that contains non-numerical values is encoded.
        In addition, every binary column will be encoded too.
        These are those columns which contain only 1's and 0's (could be either integers or floats).

    Available encodings are:
        1. one-hot (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
        2. label (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

    Args:
        edata: Central data object.
        autodetect: Whether to autodetect categorical values that will be encoded.
        encodings: Only needed if autodetect set to False.
                   A dict containing the encoding mode and categorical name for the respective column
                   or the specified encoding that will be applied to all columns.
        layer: The layer to encode.

    Returns:
        A data object with the encoded values in `.X` if `layer` is `None`, otherwise a data object with the encoded values in `layer`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> edata_encoded = ep.pp.encode(edata, autodetect=True, encodings="one-hot")

        >>> # Example using custom encodings per columns:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> # encode col1 and col2 using label encoding and encode col3 using one hot encoding
        >>> edata_encoded = ep.pp.encode(
        ...     edata, autodetect=False, encodings={"label": ["col1", "col2"], "one-hot": ["col3"]}
        ... )
    """
    X = edata.X if layer is None else edata.layers[layer]

    if not isinstance(edata, AnnData) and not isinstance(edata, EHRData):
        raise ValueError(f"Cannot encode object of type {type(edata)}. Can only encode AnnData or EHRData objects!")

    if isinstance(encodings, str) and not autodetect:
        raise ValueError("Passing a string for parameter encodings is only possible when using autodetect=True!")
    elif autodetect and not isinstance(encodings, str | type(None)):
        raise ValueError(
            f"Setting encode mode with autodetect=True only works by passing a string (encode mode name) or None not {type(encodings)}!"
        )

    if "original" not in edata.layers.keys():
        edata.layers["original"] = X.copy()

    # autodetect categorical values based on feature types stored in edata.var[FEATURE_TYPE_KEY]
    if autodetect:
        categoricals_names = _get_var_indices_for_type(edata, CATEGORICAL_TAG)

        if "encoding_mode" in edata.var.keys():
            if edata.var["encoding_mode"].isnull().values.any():
                not_encoded_features = edata.var["encoding_mode"].isna().index
                categoricals_names = [
                    _categorical for _categorical in categoricals_names if _categorical in not_encoded_features
                ]
            else:
                logger.warning(
                    "The current AnnData/EHRData object has been already encoded. Returning original AnnData/EHRData object!"
                )
                return edata

        # filter out categorical columns, that are already stored numerically
        df_edata = ed.io.to_pandas(edata, layer=layer)
        categoricals_names = [
            feat for feat in categoricals_names if not np.all(df_edata[feat].apply(type).isin([int, float]))
        ]

        # no columns were detected, that would require an encoding (e.g. non-numerical columns)
        if not categoricals_names:
            logger.warning(
                "Detected no columns that need to be encoded. Leaving passed EHRData/AnnData object unchanged."
            )
            return edata
        # update obs with the original categorical values
        updated_obs = _update_obs(edata, categoricals_names, layer=layer)

        encoded_x = None
        encoded_var_names = edata.var_names.to_list()
        unencoded_var_names = edata.var_names.to_list()
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
                edata=edata,
                X=encoded_x,
                layer=layer,
                updated_obs=updated_obs,
                var_names=encoded_var_names,
                unencoded_var_names=unencoded_var_names,
                categoricals=categoricals_names,
                progress=progress,
                task=task,
            )
            progress.update(task, description="Updating layer originals ...")

            # update layer content with the latest categorical encoding and the old other values
            updated_layer = _update_layer_after_encoding(
                edata.layers["original"],
                encoded_x,
                encoded_var_names,
                edata.var_names.to_list(),
                categoricals_names,
            )
            progress.update(task, description=f"[bold blue]Finished {encodings} of autodetected columns.")

            # copy non-encoded columns, and add new tag for encoded columns. This is needed to track encodings
            new_var = pd.DataFrame(index=encoded_var_names)
            new_var[FEATURE_TYPE_KEY] = edata.var[FEATURE_TYPE_KEY].copy()
            new_var.loc[new_var.index.str.contains("ehrapycat"), FEATURE_TYPE_KEY] = CATEGORICAL_TAG

            new_var["unencoded_var_names"] = unencoded_var_names

            new_var["encoding_mode"] = [encodings if var in categoricals_names else None for var in unencoded_var_names]

            # TODO: ehrdata v0.0.5 and newer allow to pass layers and X to constructor
            encoded_edata = _cast_adata_to_match_data_type(
                AnnData(
                    X=encoded_x,
                    obs=updated_obs,
                    var=new_var,
                    uns=edata.uns.copy(),
                ),
                edata,
            )

            encoded_edata.layers["original"] = updated_layer

    # user passed categorical values with encoding mode for each of them
    else:
        # re-encode data
        if "encoding_mode" in edata.var.keys():
            encodings = _reorder_encodings(edata, encodings)  # type: ignore
            edata = _undo_encoding(edata, layer=layer)

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
        elif any(
            _categorical in edata.var_names[edata.var[FEATURE_TYPE_KEY] == NUMERIC_TAG] for _categorical in categoricals
        ):
            logger.warning(
                "At least one of passed column names seems to have numerical dtype. In general it is not recommended "
                "to encode numerical columns!"
            )

        updated_obs = _update_obs(edata, categoricals, layer)

        encoding_mode = {}
        encoded_x = None
        encoded_var_names = edata.var_names.to_list()
        unencoded_var_names = edata.var_names.to_list()
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
                    edata=edata,
                    X=encoded_x,
                    layer=layer,
                    updated_obs=updated_obs,
                    var_names=encoded_var_names,
                    unencoded_var_names=unencoded_var_names,
                    categoricals=encodings[encoding],  # type: ignore
                    progress=progress,
                    task=task,
                )

                for _categorical in encodings[encoding]:  # type: ignore
                    _categorical = [_categorical] if isinstance(_categorical, str) else _categorical  # type: ignore
                    for column_name in _categorical:
                        # get idx of column in unencoded_var_names
                        indices = [i for i, var in enumerate(unencoded_var_names) if var == column_name]
                        encoded_var = [encoded_var_names[idx] for idx in indices]
                        for var in encoded_var:
                            encoding_mode[var] = encoding

        # update original layer content with the new categorical encoding and the old other values
        updated_layer = _update_layer_after_encoding(
            edata.layers["original"],
            encoded_x,
            encoded_var_names,
            edata.var_names.to_list(),
            categoricals,
        )

        # copy non-encoded columns, and add new tag for encoded columns. This is needed to track encodings
        new_var = pd.DataFrame(index=encoded_var_names)

        new_var[FEATURE_TYPE_KEY] = edata.var[FEATURE_TYPE_KEY].copy()
        new_var.loc[new_var.index.str.contains("ehrapycat"), FEATURE_TYPE_KEY] = CATEGORICAL_TAG

        new_var["unencoded_var_names"] = unencoded_var_names

        # update encoding mode in var, keeping already annotated columns
        if "encoding_mode" in edata.var.keys():
            encoding_mode.update({key: value for key, value in edata.var["encoding_mode"].items() if value is not None})
        new_var["encoding_mode"] = [None] * len(new_var)
        for _categorical in encoding_mode.keys():
            new_var.loc[_categorical, "encoding_mode"] = encoding_mode[_categorical]

        try:
            # TODO: ehrdata v0.0.5 and newer allow to pass layers and X to constructor
            encoded_edata = _cast_adata_to_match_data_type(
                AnnData(
                    X=encoded_x,
                    obs=updated_obs,
                    var=new_var,
                    uns=edata.uns.copy(),
                ),
                edata,
            )

            encoded_edata.layers["original"] = updated_layer

        # if the user did not pass every non-numerical column for encoding, an Anndata object cannot be created
        except ValueError:
            raise EHRDataCreationError(
                "Creation of EHRData object failed. Ensure that you passed all non numerical, "
                "categorical values for encoding!"
            ) from None

    if layer is None:
        encoded_edata.X = encoded_edata.X.astype(np.float32)
    else:
        encoded_edata.layers[layer] = encoded_edata.X.astype(np.float32)
        encoded_edata.X = None

    return encoded_edata


def _one_hot_encoding(
    edata: EHRData | AnnData,
    *,
    X: np.ndarray | None,
    layer: str | None,
    updated_obs: pd.DataFrame,
    var_names: list[str],
    unencoded_var_names: list[str],
    categoricals: list[str],
    progress: Progress,
    task,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Encode categorical columns using one hot encoding.

    Args:
        edata: Central data object.
        X: Current (encoded) X
        layer: The layer to encode.
        updated_obs: A copy of the original obs where the original categorical values are stored that will be encoded
        var_names: Names of variables to consider.
        unencoded_var_names: Unencoded var names.
        categoricals: The name of the categorical columns to be encoded
        progress: Rich Progress object.
        task: Rich Task object.

    Returns:
        Encoded new X and the corresponding new var names
    """
    original_values = _initial_encoding(updated_obs, categoricals)
    progress.update(task, description="[bold blue]Running one-hot encoding on passed columns ...")

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(original_values)
    categorical_prefixes = [
        f"ehrapycat_{category}_{str(suffix).strip()}"
        for idx, category in enumerate(categoricals)
        for suffix in encoder.categories_[idx]
    ]
    unencoded_prefixes = [category for idx, category in enumerate(categoricals) for suffix in encoder.categories_[idx]]
    transformed = encoder.transform(original_values)
    # X is None if this is the first encoding "round" -> take the former X
    if X is None:
        X = edata.X if layer is None else edata.layers[layer]
    progress.advance(task, 1)
    progress.update(task, description="[blue]Updating X and var ...")

    temp_x, temp_var_names, unencoded_var_names = _update_encoded_data(
        X, transformed, var_names, categorical_prefixes, categoricals, unencoded_prefixes, unencoded_var_names
    )
    progress.update(task, description="[blue]Finished one-hot encoding.")

    return temp_x, temp_var_names, unencoded_var_names


def _label_encoding(
    edata: EHRData | AnnData,
    *,
    X: np.ndarray | None,
    layer: str | None,
    updated_obs: pd.DataFrame,
    var_names: list[str],
    unencoded_var_names: list[str],
    categoricals: list[str],
    progress: Progress,
    task,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Encode categorical columns using label encoding.

    Args:
        edata: Central data object.
        X: Current (encoded) X.
        layer: The layer to encode.
        updated_obs: A copy of the original obs where the original categorical values are stored that will be encoded.
        var_names: Var names of current data object.
        unencoded_var_names: Unencoded var names.
        categoricals: The name of the categorical columns, that need to be encoded.
        progress: Rich Progress object.
        task: Rich Task.

    Returns:
        Encoded new X and the corresponding new var names.
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
    category_prefixes = [f"ehrapycat_{_categorical}" for _categorical in categoricals]
    # X is None if this is the first encoding "round" -> take the former X
    if X is None:
        X = edata.X if layer is None else edata.layers[layer]

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

    Returns:
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
        var_names: Var names of current data object
        categorical_prefixes: The name(s) of the encoded column(s)
        categoricals: The categorical values that were encoded recently
        unencoded_prefixes: The unencoded names of the categorical columns that were encoded
        unencoded_var_names: The unencoded names of the var columns that were encoded.

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
    array = np.array([obs[categoricals[i]].to_numpy() for i in range(len(categoricals))]).transpose()

    return array


def _undo_encoding(
    edata: EHRData | AnnData,
    layer: str | None = None,
) -> EHRData | None:
    """Undo the current encodings applied to all columns in X. This currently resets the AnnData object to its initial state.

    Args:
        edata: Central data object.
        layer: The layer to operate on.

    Returns:
        A (partially) encoding reset data object
    """
    # get all encoded features
    categoricals = _get_encoded_features(edata)

    # get all columns that should be stored in obs only
    columns_obs_only = [column_name for column_name in list(edata.obs.columns) if column_name not in categoricals]

    transformed = _initial_encoding(edata.obs, categoricals)
    temp_x, temp_var_names = _delete_all_encodings(edata, layer=layer)
    new_x = np.hstack((transformed, temp_x)) if temp_x is not None else transformed
    new_var_names = categoricals + temp_var_names if temp_var_names is not None else categoricals

    # only keep columns in obs that were stored in obs only -> delete every encoded column from obs
    new_obs = edata.obs[columns_obs_only]

    var = pd.DataFrame(index=new_var_names)
    var[FEATURE_TYPE_KEY] = [
        edata.var.loc[edata.var["unencoded_var_names"] == unenc_var_name, FEATURE_TYPE_KEY].unique()[0]
        for unenc_var_name in new_var_names
    ]
    # TODO: ehrdata v0.0.5 and newer allow to pass layers and X to constructor
    edata = _cast_adata_to_match_data_type(
        AnnData(
            new_x,
            obs=new_obs,
            var=var,
            uns=OrderedDict(),
        ),
        edata,
    )
    if layer is not None:
        edata.layers[layer] = new_x.copy()
        edata.X = None
    edata.layers["original"] = new_x.copy()

    return edata


def _delete_all_encodings(edata: EHRData | AnnData, layer: str | None) -> tuple[np.ndarray | None, list | None]:
    """Delete all encoded columns and keep track of their indices.

    Args:
        edata: Central data object.
        layer: The layer to operate on.

    Returns:
        A temporary X were all encoded columns are deleted and all var_names of unencoded columns.
    """
    X = edata.X if layer is None else edata.layers[layer]
    var_names = list(edata.var_names)
    if X is not None and var_names is not None:
        idx = 0
        for var in var_names:
            if not var.startswith("ehrapycat"):
                break
            idx += 1
        # case: only encoded columns were found
        if idx == len(var_names):
            return None, None
        # don't need to consider case when no encoded columns are there, since undo_encoding would not run anyways in this case

        return X[:, idx:].copy(), var_names[idx:]
    return None, None


def _reorder_encodings(edata: EHRData | AnnData, new_encodings: dict[str, list[list[str]] | list[str]]):
    """Reorder the encodings and update which column will be encoded using which mode.

    Args:
        edata: Central data object.
        new_encodings: The new encodings passed by the user (might affect encoded as well as previously non encoded columns)

    Returns:
        An updated encoding scheme
    """
    latest_encoded_columns = sum(new_encodings.values(), [])

    # check for duplicates and raise an error if any
    if len(set(latest_encoded_columns)) != len(latest_encoded_columns):
        logger.error(
            "Reencoding EHRData/AnnData object failed. You have at least one duplicate in your encodings. A column "
            "cannot be encoded at the same time using different encoding modes!"
        )
        raise DuplicateColumnEncodingError

    encodings = {}
    for encode_mode in available_encodings:
        encoded_categoricals_with_mode = (
            edata.var.loc[edata.var["encoding_mode"] == encode_mode, "unencoded_var_names"].unique().tolist()
        )

        encodings[encode_mode] = new_encodings[encode_mode] if encode_mode in new_encodings.keys() else []
        # add all columns that were encoded with the current mode before and are not reencoded
        encodings[encode_mode] += [
            _categorical
            for _categorical in encoded_categoricals_with_mode
            if _categorical not in latest_encoded_columns
        ]

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


def _update_obs(edata: EHRData | AnnData, categorical_names: list[str], layer: str | None = None) -> pd.DataFrame:
    """Add the original categorical values to obs.

    Args:
        edata: Central data object.
        categorical_names: Name of each categorical column
        layer: The layer to operate on.

    Returns:
        Updated obs with the original categorical values added
    """
    X = edata.X if layer is None else edata.layers[layer]
    updated_obs = edata.obs.copy()
    for idx, var_name in enumerate(edata.var_names):
        if var_name in updated_obs.columns:
            continue
        elif var_name in categorical_names:
            updated_obs[var_name] = X[::, idx : idx + 1].flatten()
            # note: this will count binary columns (0 and 1 only) as well
            # needed for writing to .h5ad files
            if set(pd.unique(updated_obs[var_name])).issubset({False, True, np.nan}):
                updated_obs[var_name] = updated_obs[var_name].astype("bool")
    # get all non bool object columns and cast them to category dtype
    object_columns = list(updated_obs.select_dtypes(include="object").columns)
    updated_obs[object_columns] = updated_obs[object_columns].astype("category")
    logger.info(f"The original categorical values `{categorical_names}` were added to obs.")

    return updated_obs


def _get_encoded_features(edata: EHRData | AnnData) -> list[str]:
    """Get all encoded features in an data object.

    Args:
        edata: Central data object.

    Returns:
        List of all unencoded names of features that were encoded
    """
    encoded_features = [
        unencoded_feature
        for enc_mode, unencoded_feature in edata.var[["encoding_mode", "unencoded_var_names"]].values
        if enc_mode is not None and not pd.isna(enc_mode)
    ]
    return list(set(encoded_features))


class EHRDataCreationError(ValueError):
    pass


class DuplicateColumnEncodingError(ValueError):
    pass
