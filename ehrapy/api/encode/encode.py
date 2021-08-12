import sys
from itertools import chain
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from anndata import AnnData
from category_encoders import CountEncoder
from rich import print
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ehrapy.api.encode._categoricals import _detect_categorical_columns


class Encoder:
    """The main encoder for the initial read AnnData object providing various encoding solutions for
    non numerical or categorical data"""

    available_encode_modes = {"one_hot_encoding", "label_encoding", "count_encoding"}

    @staticmethod
    def encode(
        ann_data: AnnData, autodetect: bool = False, categoricals_encode_mode: Dict = None
    ) -> AnnData:  # TODO specify Dict types
        """Encode the initial read AnnData object. Categorical values could be either passed via parameters or autodetected.

        Available encodings are:

        1. one-hot encoding
        2. label encoding
        3. count encoding

        Args:
            ann_data: The inital AnnData object parsed by the :class: DataReader
            autodetect: Autodetection of categorical values (default: False)
            categoricals_encode_mode: Only needed if autodetect set to False. A dict containing the categorical name
            and the encoding mode for the respective column
        """
        # TODO Add some links or descriptions to the available encodings in the docstring
        # autodetect categorical values, which could lead to more categoricals
        if autodetect:
            ann_data.uns["categoricals"] = _detect_categorical_columns(ann_data.X, ann_data.var_names)
            categorical_names = ann_data.uns["categoricals"]["categorical_encoded"]
            # TODO raise custom warning instead and proceed?
            if "current_encodings" in ann_data.uns.keys():
                print(
                    "[bold yellow]The current data has already been encoded."
                    "It is not recommended to use autodetect with already encoded data."
                )
                sys.exit(1)
            Encoder.add_categories_to_obs(ann_data, categorical_names)
            Encoder.add_categories_to_uns(ann_data, categorical_names)

            encoded_x = None
            encoded_var_names = ann_data.var_names.to_list()

            encoded_x, encoded_var_names = Encoder.one_hot_encoding(
                ann_data,
                encoded_x,
                encoded_var_names,
                categorical_names,
            )

            # update layer content with the latest categorical encoding and the old other values
            updated_layer = Encoder.update_layer_after_encode(
                ann_data.layers["original"],
                encoded_x,
                encoded_var_names,
                ann_data.var_names.to_list(),
                categorical_names,
            )
            encoded_ann_data = AnnData(
                encoded_x,
                obs=ann_data.obs.copy(),
                var=dict(var_names=encoded_var_names),
                uns=ann_data.uns.copy(),
                layers={"original": updated_layer},
            )
            encoded_ann_data.uns["current_encodings"] = {
                categorical: "one_hot_encoding" for categorical in categorical_names
            }

        # user passed categorical values with encoding mode for each of them
        else:
            ann_data.uns["categoricals_encoded_with_mode"] = categoricals_encode_mode
            categoricals = list(chain(*categoricals_encode_mode.values()))

            # ensure no categorical column gets encoded twice
            if len(categoricals) != len(set(categoricals)):
                raise ValueError(
                    "The categorical column names given contain at least one duplicate column. "
                    "Check the column names to ensure that no column is encoded twice!"
                )
            Encoder.add_categories_to_obs(ann_data, categoricals)
            Encoder.add_categories_to_uns(ann_data, categoricals)
            current_encodings = (
                {} if "current_encodings" not in ann_data.uns.keys() else ann_data.uns["current_encodings"]
            )
            encoded_x = None
            encoded_var_names = ann_data.var_names.to_list()

            for encoding_mode in categoricals_encode_mode.keys():
                if encoding_mode not in Encoder.available_encode_modes:
                    raise ValueError(
                        f"Unknown encoding mode {encoding_mode}. Please provide one of the following encoding modes:\n"
                        f"{Encoder.available_encode_modes}"
                    )
                encode_mode_switcher = {
                    "one_hot_encoding": Encoder.one_hot_encoding,
                    "label_encoding": Encoder.label_encoding,
                    "count_encoding": Encoder.count_encoding,
                }
                # perform the actual encoding
                encoded_x, encoded_var_names = encode_mode_switcher[encoding_mode](
                    ann_data, encoded_x, encoded_var_names, categoricals_encode_mode[encoding_mode]
                )
                # update encoding history in uns
                for categorical in categoricals_encode_mode[encoding_mode]:
                    current_encodings[categorical] = encoding_mode

            # update original layer content with the new categorical encoding and the old other values
            updated_layer = Encoder.update_layer_after_encode(
                ann_data.layers["original"],
                encoded_x,
                encoded_var_names,
                ann_data.var_names.to_list(),
                categoricals,
            )
            try:
                encoded_ann_data = AnnData(
                    encoded_x,
                    obs=ann_data.obs.copy(),
                    var=dict(var_names=encoded_var_names),
                    uns=ann_data.uns.copy(),
                    layers={"original": updated_layer},
                )
                # update current encodings in uns
                encoded_ann_data.uns["current_encodings"] = current_encodings

            # if the user did not pass every non numerical column for encoding, a Anndata object cannot be created
            # TODO can this be checked at an earlier time point?
            except ValueError:
                print(
                    "[bold red]Creation of AnnData object failed. "
                    "Ensure that you passed all non numerical, categorical values for encoding!"
                )
                sys.exit(1)
        del ann_data.obs  # TODO does this even make a difference? the scope if this is over anyways
        del ann_data.X

        return encoded_ann_data

    @staticmethod
    def one_hot_encoding(
        adata: AnnData,
        X: Optional[np.ndarray],
        var_names: List[str],
        categories: List[str],
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
        original_values = Encoder._init_encoding(adata, categories)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(original_values)
        cat_prefixes = [
            f"ehrapycat_{category}_{str(suffix).strip()}"
            for idx, category in enumerate(categories)
            for suffix in encoder.categories_[idx]
        ]
        transformed = encoder.transform(original_values)
        # X is None, if this is the first encoding "round", so take the "old" X
        if X is None:
            X = adata.X
        temp_x, temp_var_names = Encoder._update_encoded_data(X, transformed, var_names, cat_prefixes, categories)

        return temp_x, temp_var_names

    @staticmethod
    def label_encoding(
        adata: AnnData,
        X: Optional[np.ndarray],
        var_names: List[str],
        categories: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Encode categorical columns using label encoding.

        Args:
            adata: The current AnnData object
            X: Current (encoded) X
            var_names: Var names of current AnnData object
            categories: The name of the categorical columns, that need to be encoded

        Returns:
            Encoded new X and the corresponding new var names
        """
        original_values = Encoder._init_encoding(adata, categories)
        # label encoding expects input array to be 1D, so iterate over all columns and encode them one by one
        for idx in range(original_values.shape[1]):
            label_encoder = LabelEncoder()
            row_vec = original_values[:, idx : idx + 1].ravel()  # type: ignore
            label_encoder.fit(row_vec)
            transformed = label_encoder.transform(row_vec)
            # need a column vector instead of row vector
            original_values[:, idx : idx + 1] = transformed[..., None]
        category_prefixes = [f"ehrapycat_{category}" for category in categories]
        # X is None, if this is the first encoding "round", so take the "old" X
        if X is None:
            X = adata.X
        temp_x, temp_var_names = Encoder._update_encoded_data(
            X, original_values, var_names, category_prefixes, categories
        )

        return temp_x, temp_var_names

    @staticmethod
    def count_encoding(
        adata: AnnData,
        X: Optional[np.ndarray],
        var_names: List[str],
        categoricals: List[str],
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
        original_values = Encoder._init_encoding(adata, categoricals)

        # returns a pandas dataframe per default, but numpy array is needed
        count_encoder = CountEncoder(return_df=False)
        count_encoder.fit(original_values)
        cat_prefix = [f"ehrapycat_{cat}" for cat in categoricals]
        transformed = count_encoder.transform(original_values)
        # X is None, if this is the first encoding "round", so take the "old" X
        if X is None:
            X = adata.X  # noqa: N806
        temp_x, temp_var_names = Encoder._update_encoded_data(X, transformed, var_names, cat_prefix, categoricals)

        return temp_x, temp_var_names

    @staticmethod
    def update_layer_after_encode(
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
            # this case is needed, when there are one or more numerical (but categorical) columns, that was not encoded yet
            if col_name not in categories:
                idx_list.append(idx + old_cat_stop_index)
        # slice old original layer using the selector
        old_layer_view = old_layer[:, idx_list]
        # get all encoded categoricals of X
        encoded_categoricals = new_x[:, :new_cat_stop_index]
        # horizontally stack all encoded categoricals and the remaining "old original values"
        updated_layer = np.hstack((encoded_categoricals, old_layer_view))
        del old_layer

        return updated_layer

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
            X: Current (old) X
            transformed: The encoded (transformed) categorical column
            var_names: Var names of current AnnData object
            categorical_prefixes: The name(s) of the encoded column(s)
            categoricals: The categorical values that were encoded recently
        Returns:
            Encoded new X and the corresponding new var names
        """
        idx = Encoder._get_categories_old_indices(var_names, categoricals)
        # delete the original categorical column
        del_cat_column_x = np.delete(X, list(idx), 1)
        # create the new, encoded X
        temp_x = np.hstack((transformed, del_cat_column_x))
        # delete old categorical name
        var_names = [col_name for col_idx, col_name in enumerate(var_names) if col_idx not in idx]
        temp_var_names = categorical_prefixes + var_names

        return temp_x, temp_var_names

    @staticmethod
    def _init_encoding(
        adata: AnnData,
        categoricals: List[str],
    ) -> np.ndarray:
        """Get all original values for every categorical that needs to be encoded (again)

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
    def _get_categories_old_indices(old_var_names: List[str], encoded_categories: List[str]) -> Set[int]:
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
                if any(old_var_name[10:].startswith(cat) for cat in category_set):
                    idx_list.add(idx)

        return idx_list

    @staticmethod
    def add_categories_to_obs(ann_data: AnnData, categorical_names: List[str]) -> None:
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
    def add_categories_to_uns(ann_data: AnnData, categorical_names: List[str]) -> None:
        """Add the original categorical values to uns.

        Args:
            ann_data: The current AnnData object
            categorical_names: Name of each categorical column
        """
        is_init = "original_values_categoricals" in ann_data.uns.keys()
        ann_data.uns["original_values_categoricals"] = (
            {} if not is_init else ann_data.uns["original_values_categoricals"].copy()
        )

        for idx, var_name in enumerate(ann_data.var_names):
            if is_init and var_name in ann_data.uns["original_values_categoricals"]:
                continue
            elif var_name in categorical_names:
                ann_data.uns["original_values_categoricals"][var_name] = ann_data.X[::, idx : idx + 1]
