import sys
from itertools import chain
from typing import List, Optional, Set, Tuple

import numpy as np
from anndata import AnnData
from category_encoders import CountEncoder, HashingEncoder, SumEncoder
from rich import print
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ehrapy.api.encode._categoricals import _detect_categorical_columns


class Encoder:
    """The main encoder for the initial read AnnData object providing various encoding solutions for
    non numerical or categorical data"""

    available_encode_modes = {"one_hot_encoding", "label_encoding", "count_encoding", "hash_encoding", "sum_encoding"}

    @staticmethod
    def encode(ann_data: AnnData, autodetect=False, categoricals_encode_mode=None) -> AnnData:
        """Encode the inital read AnnData object. Categorical values could be either passed via parameters or autodetected.

        Args:
            ann_data: The inital AnnData object parsed by the :class: Datareader
            autodetect: Autodetection of categorical values (default: False)
            categoricals_encode_mode: Only needed if autodetect set to False. A dict containing the categorical name
            and the encoding mode for the respective column
        """
        # autodetect categorical values, which could lead to more categoricals
        if autodetect:
            ann_data.uns["categoricals"] = _detect_categorical_columns(ann_data.X, ann_data.var_names)
            cat_names = ann_data.uns["categoricals"]["categorical_encoded"]
            Encoder.add_cats_to_obs(ann_data, cat_names)
            Encoder.add_cats_to_uns(ann_data, cat_names)

            encoded_x = None
            encoded_var_names = ann_data.var_names.to_list()

            encoded_x, encoded_var_names = Encoder.one_hot_encoding(
                ann_data,
                encoded_x,
                encoded_var_names,
                cat_names,
            )

            # update layer content with the latest categorical encoding and the old other values
            updated_layer = Encoder.update_layer_after_encode(
                ann_data.layers["original"], encoded_x, encoded_var_names, ann_data.var_names.to_list(), cat_names
            )
            encoded_ann_data = AnnData(
                encoded_x,
                obs=ann_data.obs.copy(),
                var=dict(var_names=encoded_var_names),
                layers={"original": updated_layer},
            )

        # user passed categorical values with encoding mode for each
        else:
            ann_data.uns["categoricals_encoded_with_mode"] = categoricals_encode_mode
            categoricals = list(chain(*categoricals_encode_mode.values()))
            # ensure no categorical column gets encoded twice
            if len(categoricals) != len(set(categoricals)):
                raise ValueError("The categorical column names given contain at least one duplicate column. Check the column names "
                                 "to ensure not encoding a column twice!")
            Encoder.add_cats_to_obs(ann_data, categoricals)
            Encoder.add_cats_to_uns(ann_data, categoricals)
            current_encodings = {"current_encodings": {}} if "current_encodings" not in ann_data.uns.keys() else ann_data.uns["current_encodings"]
            encoded_x = None
            encoded_var_names = ann_data.var_names.to_list()

            for encoding_mode in categoricals_encode_mode.keys():
                if encoding_mode not in Encoder.available_encode_modes:
                    raise ValueError(
                        f"Unknown encoding mode {encoding_mode}. Please provide one of the following encoding modes:\n"
                        f"{Encoder.available_encode_modes}"
                    )
                # check, whether encoding mode encodes multiple columns together
                # this will be important, when a column gets encoded again, since all the other encoded columns need to be encoded again as well
                is_multiple_encoding = Encoder.is_multiple_encode_mode(encoding_mode)
                encode_mode_switcher = {
                    "one_hot_encoding": Encoder.one_hot_encoding,
                    "label_encoding": Encoder.label_encoding,
                    "count_encoding": Encoder.count_encoding,
                    # TODO: Hash and sum encoding hashes/sum several columns together, to be shape consistent its necessary (if encode again)
                    # TODO: to encode every column previously hashed/summed again if any of them gets anecoded again (throw error if not)
                    "hash_encoding": Encoder.hash_encoding,
                    "sum_encoding": Encoder.sum_encoding,
                }
                encoded_x, encoded_var_names = encode_mode_switcher[encoding_mode](
                    ann_data, encoded_x, encoded_var_names, categoricals_encode_mode[encoding_mode]
                )
                # update encoding history in uns
                for categorical in categoricals_encode_mode[encoding_mode]:
                    current_encodings["current_encodings"][categorical] = (encoding_mode, [] if not is_multiple_encoding else categoricals_encode_mode[encoding_mode])

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
                encoded_ann_data.uns['current_encodings'] = current_encodings

            # if the user did not pass at least every non numerical column for encoding, a Anndata object cannot be created
            except ValueError:
                print(
                    "[bold red]Creation of AnnData object failed. Ensure that you passed at least all non numerical, categorical"
                    "values for encoding!"
                )
                sys.exit(1)
        del ann_data.obs
        del ann_data.X
        return encoded_ann_data

    @staticmethod
    def one_hot_encoding(
        ann_data: AnnData,
        X: Optional[np.ndarray],  # noqa: N803
        var_names: List[str],
        cats: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Encode categorical columns using one hot encoding.

        Args:
            ann_data: The current AnnData object
            X: Current (encoded) X
            var_names: Var names of current AnnData object
            cats: The name of the categorical column to be encoded
        """
        arr = Encoder.init_encoding(ann_data, cats)  # noqa: N806
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(arr)
        cat_prefixes = [
            f"ehrapycat_{cat}_{suffix.strip()}" for idx, cat in enumerate(cats) for suffix in enc.categories_[idx]
        ]
        transformed = enc.transform(arr)
        # X is None, if this is the first encoding "round", so take the "old" X
        if X is None:
            X = ann_data.X  # noqa: N806
        temp_x, temp_var_names = Encoder.update_encoded_data(X, transformed, var_names, cat_prefixes, cats)

        return temp_x, temp_var_names

    @staticmethod
    def label_encoding(
        ann_data: AnnData,
        X: Optional[np.ndarray],  # noqa: N803
        var_names: List[str],
        cats: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Encode categorical columns using label encoding.

        Args:
            ann_data: The current AnnData object
            X: Current (encoded) X
            var_names: Var names of current AnnData object
            cats: The name of the categorical columns, that need to be encoded
        """
        arr = Encoder.init_encoding(ann_data, cats)

        label_encoder = LabelEncoder()
        row_vec = arr.ravel()
        label_encoder.fit(row_vec)
        cat_prefixes = [f"ehrapycat_{cat}" for cat in cats]
        transformed = label_encoder.transform(row_vec)
        # need a column vector instead of row vector
        transformed = transformed[..., None]
        # X is None, if this is the first encoding "round", so take the "old" X
        if X is None:
            X = ann_data.X  # noqa: N806
        temp_x, temp_var_names = Encoder.update_encoded_data(X, transformed, var_names, cat_prefixes, cats)

        return temp_x, temp_var_names

    @staticmethod
    def count_encoding(
        ann_data: AnnData,
        X: Optional[np.ndarray],  # noqa: N803
        var_names: List[str],
        cats: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Encode categorical column using count encoding.

        Args:
            ann_data: The current AnnData object
            X: Current (encoded) X
            var_names: Var names of current AnnData object
            cats: The name of the categorical columns, that need to be encoded
        """
        arr = Encoder.init_encoding(ann_data, cats)

        # returns a pandas dataframe per default, but numpy array is needed
        count_encoder = CountEncoder(return_df=False)
        count_encoder.fit(arr)
        cat_prefix = [f"ehrapycat_{cat}" for cat in cats]
        transformed = count_encoder.transform(arr)
        # X is None, if this is the first encoding "round", so take the "old" X
        if X is None:
            X = ann_data.X  # noqa: N806
        temp_x, temp_var_names = Encoder.update_encoded_data(X, transformed, var_names, cat_prefix, cats)

        return temp_x, temp_var_names

    @staticmethod
    def hash_encoding(
        ann_data: AnnData,
        X: Optional[np.ndarray],  # noqa: N803
        var_names: List[str],
        cats: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Encode categorical columns using hash encoding.

        Args:
            ann_data: The current AnnData object
            X: Current (encoded) X
            var_names: Var names of current AnnData object
            cats: The name of the categorical columns, that need to be encoded
        """
        arr = Encoder.init_encoding(ann_data, cats)

        # returns a pandas dataframe per default, but numpy array is needed
        hash_encoder = HashingEncoder(return_df=False)
        hash_encoder.fit(arr)
        transformed = hash_encoder.transform(arr)
        cat_prefixes = [f"ehrapycat_{cat}_{feature}" for cat in cats for feature in hash_encoder.get_feature_names()]
        # X is None, if this is the first encoding "round", so take the "old" X
        if X is None:
            X = ann_data.X  # noqa: N806
        temp_x, temp_var_names = Encoder.update_encoded_data(X, transformed, var_names, cat_prefixes, cats)

        return temp_x, temp_var_names

    @staticmethod
    def sum_encoding(
        ann_data: AnnData,
        X: Optional[np.ndarray],  # noqa: N803
        var_names: List[str],
        cats: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Encode categorical columns using sum encoding.

        Args:
            ann_data: The current AnnData object
            X: Current (encoded) X
            var_names: Var names of current AnnData object
            cats: The name of the categorical columns, that need to be encoded
        """
        arr = Encoder.init_encoding(ann_data, cats)

        # returns a pandas dataframe per default, but numpy array is needed
        sum_encoder = SumEncoder(return_df=False)
        sum_encoder.fit(arr)
        transformed = sum_encoder.transform(arr)
        cat_prefixes = [f"ehrapycat_{cat}_{feature}" for cat in cats for feature in sum_encoder.get_feature_names()]
        # X is None, if this is the first encoding "round", so take the "old" X
        if X is None:
            X = ann_data.X  # noqa: N806
        temp_x, temp_var_names = Encoder.update_encoded_data(X, transformed, var_names, cat_prefixes, cats)

        return temp_x, temp_var_names

    @staticmethod
    def update_layer_after_encode(
        old_layer: np.ndarray, new_x: np.ndarray, new_var_names: List[str], old_var_names: List[str], cats: List[str]
    ) -> np.ndarray:
        """Update the original layer containing a the initial non categorical values and the latest encoded categorials

        Args:
                old_layer: The previous "oiginal" layer
                new_x: The new encoded X
                new_var_names: The new encoded var names
                old_var_names: The previous var names
                cats: All previous categorical names
        """
        # get the index of the first column of the new encoded X, that does not store an encoded categorical
        new_cat_stop_index = next(i for i in range(len(new_var_names)) if not new_var_names[i].startswith("ehrapycat"))
        # get the index of the first column of the old encoded X, that does not store an encoded categorical
        old_cat_stop_index = next(i for i in range(len(old_var_names)) if not old_var_names[i].startswith("ehrapycat"))
        # keep track of all indices with original value columns, that are (and were) not encoded
        idx_list = []
        for idx, col_name in enumerate(old_var_names[old_cat_stop_index:]):
            # this case is needed, when there are one or more numerical (but categorical) columns, that was not encoded yet
            if col_name not in cats:
                idx_list.append(idx + old_cat_stop_index)
        # slice old original layer using the selector
        old_layer_view = old_layer[:, idx_list]
        # get all encoded categoricals of X
        encoded_cats = new_x[:, :new_cat_stop_index]
        # horizontally stack all encoded categoricals and the remaining "old original values"
        updated_layer = np.hstack((encoded_cats, old_layer_view))
        del old_layer
        return updated_layer

    @staticmethod
    def update_encoded_data(
        X: np.ndarray,  # noqa: N803
        transformed: np.ndarray,
        var_names: List[str],
        cat_prefixes: List[str],
        categoricals: List[str],
    ):
        """Update X and var_names after each encoding
        Args:
            X: Current (old) X
            transformed: The encoded (transformed) categorical column
            var_names: Var names of current AnnData object
            cat_prefixes: The name(s) of the encoded column(s)
            categoricals: The categorical values that were encoded recently
        """
        idx = Encoder.get_cat_old_indices(var_names, categoricals)
        # delete the original categorical column
        del_cat_column_x = np.delete(X, list(idx), 1)
        # create the new, encoded X
        temp_x = np.hstack((transformed, del_cat_column_x))
        # delete old categorical name
        var_names = [col_name for col_idx, col_name in enumerate(var_names) if col_idx not in idx]
        temp_var_names = cat_prefixes + var_names
        return temp_x, temp_var_names

    @staticmethod
    def init_encoding(
        ann_data: AnnData,
        cats: List[str],
    ):
        """Get all original values for every categorical, that needs to be encoded (again)
        Args:
            ann_data: The current AnnData object
            cats: All categoricals, that need to be encoded
        """
        # create numpy array from all original categorical values, that will be encoded (again)
        arr = np.array(
            [ann_data.uns["original_values_categoricals"][cats[i]].ravel() for i in range(len(cats))]
        ).transpose()

        return arr

    @staticmethod
    def get_cat_old_indices(old_var_names: List[str], encoded_cats: List[str]) -> Set[int]:
        """Get the indices of every (possibly encoded) categorical column, that belongs to
        a categorical value, that is newly encoded"""
        idx_list = set()
        cat_set = set(encoded_cats)
        for idx, old_var_name in enumerate(old_var_names):
            # if the old variable was previously unencoded (only the case for numerical categoricals)
            if old_var_name in cat_set:
                idx_list.add(idx)
            # if the old variable was already encoded
            elif old_var_name.startswith("ehrapycat_"):
                if any(old_var_name[10:].startswith(cat) for cat in cat_set):
                    idx_list.add(idx)
        return idx_list

    @staticmethod
    def add_cats_to_obs(ann_data: AnnData, cat_names: List[str]) -> None:
        """Add the original categorical values to obs.

        Args:
            ann_data: The current AnnData object
            cat_names: Name of each categorical column
        """
        for idx, var_name in enumerate(ann_data.var_names):
            if var_name in ann_data.obs.columns:
                continue
            elif var_name in cat_names:
                ann_data.obs[var_name] = ann_data.X[::, idx : idx + 1]

    @staticmethod
    def add_cats_to_uns(ann_data: AnnData, cat_names: List[str]) -> None:
        """Add the original categorical values to uns.

        Args:
            ann_data: The current AnnData object
            cat_names: Name of each categorical column
        """
        is_init = "original_values_categoricals" in ann_data.uns.keys()
        ann_data.uns["original_values_categoricals"] = (
            {} if not is_init else ann_data.uns["original_values_categoricals"].copy()
        )

        for idx, var_name in enumerate(ann_data.var_names):
            if is_init and var_name in ann_data.uns["original_values_categoricals"]:
                continue
            elif var_name in cat_names:
                ann_data.uns["original_values_categoricals"][var_name] = ann_data.X[::, idx : idx + 1]

    @staticmethod
    def is_multiple_encode_mode(encode_mode: str) -> bool:
        """

        """
        if encode_mode in {'sum_encoding', 'hash_encoding'}:
            return True
        return False
