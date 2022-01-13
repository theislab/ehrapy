import sys
from itertools import chain
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from _collections import OrderedDict
from anndata import AnnData
from category_encoders import CountEncoder
from rich import print
from rich.progress import BarColumn, Progress
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ehrapy.api.encode._categoricals import _detect_categorical_columns


class Encoder:
    """The main encoder for the initial read AnnData object providing various encoding solutions for
    non numerical or categorical data"""

    available_encodings = {"one_hot_encoding", "label_encoding", "count_encoding"}

    @staticmethod
    def encode(
        adata: AnnData, autodetect: bool = False, encodings: Dict[str, List[str]] = None
    ) -> Union[AnnData, None]:
        """Encode the initial read AnnData object. Categorical values could be either passed via parameters or autodetected.

        Available encodings are:

        1. one_hot_encoding (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
        2. label_encoding (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
        3. count_encoding (https://contrib.scikit-learn.org/category_encoders/count.html)

        Label encodes by default which is used to save initially unencoded AnnData objects.

        Args:
            adata: The inital AnnData object parsed by the :class: DataReader
            autodetect: Autodetection of categorical values (default: False)
            encodings: Only needed if autodetect set to False. A dict containing the categorical name
            and the encoding mode for the respective column

        Returns:
            An :class:`~anndata.AnnData` object with the encoded values in X
        """
        # autodetect categorical values, which could lead to more categoricals
        if autodetect:
            adata.uns["categoricals"] = _detect_categorical_columns(adata.X, adata.var_names)
            categoricals_names = adata.uns["categoricals"]["categorical_encoded"]
            if "current_encodings" in adata.uns.keys():
                print(
                    "[bold yellow]The current data has already been encoded."
                    "It is not recommended to use autodetect with already encoded data."
                    "[bold red]Aborting..."
                )
                return None
            Encoder._add_categoricals_to_obs(adata, categoricals_names)
            Encoder._add_categoricals_to_uns(adata, categoricals_names)

            encoded_x = None
            encoded_var_names = adata.var_names.to_list()

            # Label encode by default. The primary usage of this is to save unencoded AnnData objects
            encoded_x, encoded_var_names = Encoder._label_encoding(
                adata,
                encoded_x,
                encoded_var_names,
                categoricals_names,
            )

            # update layer content with the latest categorical encoding and the old other values
            updated_layer = Encoder._update_layer_after_encoding(
                adata.layers["original"],
                encoded_x,
                encoded_var_names,
                adata.var_names.to_list(),
                categoricals_names,
            )
            encoded_ann_data = AnnData(
                encoded_x,
                obs=adata.obs.copy(),
                var=dict(var_names=encoded_var_names),
                uns=adata.uns.copy(),
                layers={"original": updated_layer},
            )
            encoded_ann_data.uns["current_encodings"] = {
                categorical: "one_hot_encoding" for categorical in categoricals_names
            }

        # user passed categorical values with encoding mode for each of them
        else:
            # are all specified encodings valid?
            for encoding_mode in encodings.keys():
                if encoding_mode not in Encoder.available_encodings:
                    raise ValueError(
                        f"Unknown encoding mode {encoding_mode}. Please provide one of the following encoding modes:\n"
                        f"{Encoder.available_encodings}"
                    )

            adata.uns["categoricals_encoded_with_mode"] = encodings
            categoricals = list(chain(*encodings.values()))

            # ensure no categorical column gets encoded twice
            if len(categoricals) != len(set(categoricals)):
                raise ValueError(
                    "The categorical column names given contain at least one duplicate column. "
                    "Check the column names to ensure that no column is encoded twice!"
                )
            Encoder._add_categoricals_to_obs(adata, categoricals)
            Encoder._add_categoricals_to_uns(adata, categoricals)
            current_encodings = {} if "current_encodings" not in adata.uns.keys() else adata.uns["current_encodings"]
            encoded_x = None
            encoded_var_names = adata.var_names.to_list()

            with Progress(
                "[progress.description]{task.description}", BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%"
            ) as progress:
                task = progress.add_task(
                    "[red] Setting up encoding ...", total=sum([len(val_list) for val_list in encodings.values()])
                )
                for encoding_mode in encodings.keys():
                    encode_mode_switcher = {
                        "one_hot_encoding": Encoder._one_hot_encoding,
                        "label_encoding": Encoder._label_encoding,
                        "count_encoding": Encoder._count_encoding,
                    }
                    progress.update(task, description=f"Running {encoding_mode} ...")
                    # perform the actual encoding
                    encoded_x, encoded_var_names = encode_mode_switcher[encoding_mode](
                        adata, encoded_x, encoded_var_names, encodings[encoding_mode], progress, task
                    )

                    # update encoding history in uns
                    for categorical in encodings[encoding_mode]:
                        current_encodings[categorical] = encoding_mode
                    progress.update(task, advance=1)

            # update original layer content with the new categorical encoding and the old other values
            updated_layer = Encoder._update_layer_after_encoding(
                adata.layers["original"],
                encoded_x,
                encoded_var_names,
                adata.var_names.to_list(),
                categoricals,
            )
            try:
                encoded_ann_data = AnnData(
                    encoded_x,
                    obs=adata.obs.copy(),
                    var=dict(var_names=encoded_var_names),
                    uns=adata.uns.copy(),
                    layers={"original": updated_layer},
                )
                # update current encodings in uns
                encoded_ann_data.uns["current_encodings"] = current_encodings

            # if the user did not pass every non numerical column for encoding, a Anndata object cannot be created
            except ValueError:
                print(
                    "[bold red]Creation of AnnData object failed. "
                    "Ensure that you passed all non numerical, categorical values for encoding!"
                )
                sys.exit(1)
        del adata.obs
        del adata.X

        return encoded_ann_data

    @staticmethod
    def _one_hot_encoding(
        adata: AnnData,
        X: Optional[np.ndarray],
        var_names: List[str],
        categories: List[str],
        progress: Optional[Progress] = None,
        task=None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Encode categorical columns using one hot encoding.

        Args:
            adata: The current AnnData object
            X: Current (encoded) X
            var_names: Var names of current AnnData object
            categories: The name of the categorical columns to be encoded

        Returns:
            Encoded new X and the corresponding new var names
        """
        original_values = Encoder._initial_encoding(adata, categories)

        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(original_values)
        categorical_prefixes = [
            f"ehrapycat_{category}_{str(suffix).strip()}"
            for idx, category in enumerate(categories)
            for suffix in encoder.categories_[idx]
        ]
        transformed = encoder.transform(original_values)
        # X is None if this is the first encoding "round" -> take the former X
        if X is None:
            X = adata.X
        if progress:
            progress.update(task, description="[blue]Updating one hot encoded values ...")
        temp_x, temp_var_names = Encoder._update_encoded_data(
            X, transformed, var_names, categorical_prefixes, categories
        )

        return temp_x, temp_var_names

    @staticmethod
    def _label_encoding(
        adata: AnnData,
        X: Optional[np.ndarray],
        var_names: List[str],
        categoricals: List[str],
        progress: Optional[Progress] = None,
        task=None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Encode categorical columns using label encoding.

        Args:
            adata: The current AnnData object
            X: Current (encoded) X
            var_names: Var names of current AnnData object
            categoricals: The name of the categorical columns, that need to be encoded

        Returns:
            Encoded new X and the corresponding new var names
        """
        original_values = Encoder._initial_encoding(adata, categoricals)
        # label encoding expects input array to be 1D, so iterate over all columns and encode them one by one
        for idx in range(original_values.shape[1]):
            if progress:
                progress.update(task, description=f"[blue]Running label encoding on column {categoricals[idx]} ...")
            label_encoder = LabelEncoder()
            row_vec = original_values[:, idx : idx + 1].ravel()  # type: ignore
            label_encoder.fit(row_vec)
            transformed = label_encoder.transform(row_vec)
            # need a column vector instead of row vector
            original_values[:, idx : idx + 1] = transformed[..., None]
        category_prefixes = [f"ehrapycat_{categorical}" for categorical in categoricals]
        # X is None if this is the first encoding "round" -> take the former X
        if X is None:
            X = adata.X
        if progress:
            progress.update(task, description="[blue]Updating label encoded values ...")
        temp_x, temp_var_names = Encoder._update_encoded_data(
            X, original_values, var_names, category_prefixes, categoricals
        )

        return temp_x, temp_var_names

    @staticmethod
    def _count_encoding(
        adata: AnnData,
        X: Optional[np.ndarray],
        var_names: List[str],
        categoricals: List[str],
        progress: Optional[Progress] = None,
        task=None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Encode categorical column using count encoding.

        Args:
            adata: The current AnnData object
            X: Current (encoded) X
            var_names: Var names of current AnnData object
            categoricals: The name of the categorical columns, that need to be encoded

        Returns:
            Encoded new X and the corresponding new var names
        """
        original_values = Encoder._initial_encoding(adata, categoricals)

        # returns a pandas dataframe per default, but numpy array is needed
        count_encoder = CountEncoder(return_df=False)
        count_encoder.fit(original_values)
        category_prefix = [f"ehrapycat_{categorical}" for categorical in categoricals]
        transformed = count_encoder.transform(original_values)
        # X is None if this is the first encoding "round" -> take the former X
        if X is None:
            X = adata.X  # noqa: N806
        if progress:
            progress.update(task, description="[blue]Updating count encoded values ...")
        temp_x, temp_var_names = Encoder._update_encoded_data(X, transformed, var_names, category_prefix, categoricals)

        return temp_x, temp_var_names

    @staticmethod
    def _update_layer_after_encoding(
        old_layer: np.ndarray,
        new_x: np.ndarray,
        new_var_names: List[str],
        old_var_names: List[str],
        categories: List[str],
    ) -> np.ndarray:
        """Update the original layer containing the initial non categorical values and the latest encoded categorials

        Args:
            old_layer: The previous "oiginal" layer
            new_x: The new encoded X
            new_var_names: The new encoded var names
            old_var_names: The previous var names
            categories: All previous categorical names

        Returns
            A Numpy array containing all numericals together with all encoded categoricals
        """
        # get the index of the first column of the new encoded X, that does not store an encoded categorical
        new_cat_stop_index = next(i for i in range(len(new_var_names)) if not new_var_names[i].startswith("ehrapycat"))
        # get the index of the first column of the old encoded X, that does not store an encoded categorical
        old_cat_stop_index = next(i for i in range(len(old_var_names)) if not old_var_names[i].startswith("ehrapycat"))
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
        del old_layer

        return updated_layer.astype("float32")

    @staticmethod
    def _update_encoded_data(
        X: np.ndarray,
        transformed: np.ndarray,
        var_names: List[str],
        categorical_prefixes: List[str],
        categoricals: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Update X and var_names after each encoding

        Args:
            X: Current (former) X
            transformed: The encoded (transformed) categorical column
            var_names: Var names of current AnnData object
            categorical_prefixes: The name(s) of the encoded column(s)
            categoricals: The categorical values that were encoded recently

        Returns:
            Encoded new X and the corresponding new var names
        """
        idx = Encoder._get_categoricals_old_indices(var_names, categoricals)
        # delete the original categorical column
        del_cat_column_x = np.delete(X, list(idx), 1)
        # create the new, encoded X
        temp_x = np.hstack((transformed, del_cat_column_x))
        # delete old categorical name
        var_names = [col_name for col_idx, col_name in enumerate(var_names) if col_idx not in idx]
        temp_var_names = categorical_prefixes + var_names

        return temp_x, temp_var_names

    @staticmethod
    def _initial_encoding(
        adata: AnnData,
        categoricals: List[str],
    ) -> np.ndarray:
        """Get all original values for all categoricals that need to be encoded (again)

        Args:
            adata: The current AnnData object
            categoricals: All categoricals that need to be encoded

        Returns:
            Numpy array of all original categorial values
        """
        # create numpy array from all original categorical values, that will be encoded (again)
        array = np.array(
            [adata.uns["original_values_categoricals"][categoricals[i]].ravel() for i in range(len(categoricals))]
        ).transpose()

        return array

    @staticmethod
    def undo_encoding(
        adata: AnnData, columns: str = "all", from_cache_file: bool = False, cache_file: str = None
    ) -> Optional[AnnData]:
        """
        Undo the current encodings applied to all columns in X. This currently resets the AnnData object to its initial state.

        Args:
            adata: The AnnData object
            columns: The names of the columns to reset encoding for. Defaults to all columns.
            from_cache_file: Whether to reset all encodings by reading from a cached .h5ad file, if available.
            This resets the AnnData object to its initial state.
            cache_file: The filename of the cache file to read from

        Returns:
            A (partially) encoding reset AnnData object
        """
        if "current_encodings" not in adata.uns.keys():
            print("[bold yellow]Calling undo_encoding on unencoded AnnData object. [bold red]Aborting...")
            return None

        # get all encoded variables
        encoded_categoricals = list(adata.uns["original_values_categoricals"].keys())
        # get all columns that should be stored in obs only
        columns_obs_only = [
            column_name for column_name in list(adata.obs.columns) if column_name not in encoded_categoricals
        ]

        if from_cache_file:
            # import here to resolve circular dependency issue on module level
            from ehrapy.api.io.read import DataReader

            # read from cache file and decode it
            cached_adata = DataReader.read(cache_file)
            cached_adata.X = cached_adata.X.astype("object")
            cached_adata = DataReader._decode_cached_adata(cached_adata, columns_obs_only)
            return cached_adata
        # maybe implement a way to only reset encoding for specific columns later
        if columns == "all":
            categoricals = list(adata.uns["original_values_categoricals"].keys())
        else:
            print("[bold yellow]Currently, one can only reset encodings for all columns! [bold red]Aborting...")
            return None
        transformed = Encoder._initial_encoding(adata, categoricals)
        new_x, new_var_names = Encoder._update_encoded_data(
            adata.X, transformed, list(adata.var_names), categoricals, categoricals
        )
        # only keep columns in obs that were stored in obs only -> delete every encoded column from obs
        new_obs = adata.obs[columns_obs_only]
        del adata
        return AnnData(
            new_x,
            obs=new_obs,
            var=pd.DataFrame(index=new_var_names),
            uns=OrderedDict(),
            dtype="object",
            layers={"original": new_x.copy()},
        )

    @staticmethod
    def _get_categoricals_old_indices(old_var_names: List[str], encoded_categories: List[str]) -> Set[int]:
        """Get indices of every (possibly encoded) categorical column belonging to a newly encoded categorical value

        Args:
            old_var_names: Former variables names
            encoded_categories: Already encoded categories

        Returns:
            Set of all indices of formerly encoded categories belonging to a newly encoded categorical value
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

    @staticmethod
    def _add_categoricals_to_obs(ann_data: AnnData, categorical_names: List[str]) -> None:
        """Add the original categorical values to obs.

        Args:
            ann_data: The current AnnData object
            categorical_names: Name of each categorical column
        """
        for idx, var_name in enumerate(ann_data.var_names):
            if var_name in ann_data.obs.columns:
                continue
            elif var_name in categorical_names:
                ann_data.obs[var_name] = ann_data.X[::, idx : idx + 1]

    @staticmethod
    def _add_categoricals_to_uns(ann_data: AnnData, categorical_names: List[str]) -> None:
        """Add the original categorical values to uns.

        Args:
            ann_data: The current AnnData object
            categorical_names: Name of each categorical column
        """
        is_initial = "original_values_categoricals" in ann_data.uns.keys()
        ann_data.uns["original_values_categoricals"] = (
            {} if not is_initial else ann_data.uns["original_values_categoricals"].copy()
        )

        for idx, var_name in enumerate(ann_data.var_names):
            if is_initial and var_name in ann_data.uns["original_values_categoricals"]:
                continue
            elif var_name in categorical_names:
                ann_data.uns["original_values_categoricals"][var_name] = ann_data.X[::, idx : idx + 1]


class AlreadyEncodedWarning(UserWarning):
    pass