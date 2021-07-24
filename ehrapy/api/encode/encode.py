from anndata import AnnData
from typing import Set
from rich import print
import sys
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import OneHotEncoder

from ehrapy.api.encode._categoricals import _detect_categorical_columns


class Encoder:
    """ The main encoder for the initial read AnnData object providing various encoding solutions for
    non numerical or categorical data """

    available_encode_modes = {
        "one_hot_encoding"
    }

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

            for categorical in cat_names:
                encoded_x, encoded_var_names = Encoder.one_hot_encoding(ann_data, encoded_x, encoded_var_names, categorical)

            # update layer content with the new categorical encoding and the old other values
            updated_layer = Encoder.update_layer_after_encode(ann_data.layers['original'], encoded_x, encoded_var_names, ann_data.var_names.to_list(),
                                                              cat_names)
            encoded_ann_data = AnnData(encoded_x, obs=ann_data.obs.copy(), var=dict(var_names=encoded_var_names), layers={'original': updated_layer})

        # user passed categorical values with encoding mode for each
        else:
            ann_data.uns["categoricals_encoded_with_mode"] = categoricals_encode_mode
            cat_encode_mode_keys = categoricals_encode_mode.keys()
            Encoder.add_cats_to_obs(ann_data, cat_encode_mode_keys)
            Encoder.add_cats_to_uns(ann_data, cat_encode_mode_keys)

            encoded_x = None
            encoded_var_names = ann_data.var_names.to_list()

            for categorical in categoricals_encode_mode.keys():
                mode = categoricals_encode_mode[categorical]
                if mode not in Encoder.available_encode_modes:
                    raise ValueError(
                        f"Please provide one of the available encoding modes for categorical column {categorical}.\n"
                        f"{Encoder.available_encode_modes}"
                    )
                if mode == "one_hot_encoding":
                    encoded_x, encoded_var_names = Encoder.one_hot_encoding(ann_data, encoded_x, encoded_var_names, categorical)

            # update original layer content with the new categorical encoding and the old other values
            updated_layer = Encoder.update_layer_after_encode(ann_data.layers['original'], encoded_x, encoded_var_names, ann_data.var_names.to_list(),
                                                              cat_encode_mode_keys)

            encoded_ann_data = AnnData(encoded_x, obs=ann_data.obs.copy(), var=dict(var_names=encoded_var_names), layers={'original': updated_layer})
        del ann_data.obs
        del ann_data.X
        return encoded_ann_data

    @staticmethod
    def one_hot_encoding(ann_data: AnnData, X: Optional[np.ndarray], var_names: List[str], cat: str) -> Tuple[np.ndarray, List[str]]:
        """Encode categorical column using one hot encoding.

                Args:
                    ann_data: The current AnnData object
                    X: Current (encoded) X
                    var_names: Var names of current AnnData object
                    cat: The name of the categorical column to be encoded
        """
        try:
            idx = var_names.index(cat)
        except ValueError:
            print(f"[bold red]Could not find column {cat} in AnnData var names. Please check spelling of your "
                  f"passed categorical columns!")
            sys.exit(1)
        if X is None:
            arr = ann_data.X[::, idx:idx+1]
            x = ann_data.X
        else:
            arr = X[::, idx:idx+1]
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(arr)
        cat_prefixes = [f"ehrapycat_{cat}_{suffix}" for suffix in enc.categories_[0]]
        transformed = enc.transform(arr)
        # delete the original categorical column
        # TODO Maybe we could use resize for inplace deletion
        if X is None:
            del_cat_column_x = np.delete(x, idx, 1)
        else:
            del_cat_column_x = np.delete(X, idx, 1)
        # create the new, encoded X
        temp_x = np.hstack((transformed, del_cat_column_x))
        # delete old categorical name
        del var_names[idx]
        temp_var_names = cat_prefixes + var_names

        return temp_x, temp_var_names

    @staticmethod
    def update_layer_after_encode(old_layer: np.ndarray, new_x: np.ndarray, new_var_names: List[str], old_var_names: List[str], cats: List[str]) -> np.ndarray:
        """Update the original layer containing a the initial non categorical values and the latest encoded categorials

                Args:
                        old_layer: The previous "oiginal" layer
                        new_x: The new encoded X
                        new_var_names: The new encoded var names
                        old_var_names: The previous var names
                        cats: All previous categorical names
        """
        indices = set()
        # get the indices for all categoricals in the old original layer
        for cat in cats:
            indices.add(old_var_names.index(cat))
        # get the stop index up to which the (encoded) categoricals are stored
        stop_index = next(i for i in range(len(new_var_names)) if not new_var_names[i].startswith("ehrapycat"))
        # create a selector that filters als indices that are not categorical columns in the old original layer
        selector = [x for x in range(old_layer.shape[1]) if x not in indices]
        # slice old original layer using the selector
        old_layer_view = old_layer[:, selector]
        # get all encoded categoricals
        new_encoded_cats = new_x[:, :stop_index]
        updated_layer = np.hstack((new_encoded_cats, old_layer_view))
        return updated_layer

    @staticmethod
    def add_cats_to_obs(ann_data: AnnData, cat_names: Set[str]) -> None:
        """Add the original categorical values to obs.

                Args:
                    ann_data: The current AnnData object
                    cat_names: Name of each categorical column
        """
        for idx, var_name in enumerate(ann_data.var_names):
            if var_name in cat_names:
                ann_data.obs[var_name] = ann_data.X[::, idx:idx+1]

    @staticmethod
    def add_cats_to_uns(ann_data: AnnData, cat_names: Set[str]) -> None:
        """Add the original categorical values to uns.

                Args:
                    ann_data: The current AnnData object
                    cat_names: Name of each categorical column
        """
        ann_data.uns["original_values_categoricals"] = {}
        for idx, var_name in enumerate(ann_data.var_names):
            if var_name in cat_names:
                ann_data.uns["original_values_categoricals"][var_name] = ann_data.X[::, idx:idx + 1]
