from typing import Dict, List, Union

from anndata import AnnData

from ehrapy.api.encode.encode import Encoder


def encode(ann_data: AnnData, autodetect: bool = False, encodings: Dict[str, List[str]] = None) -> AnnData:
    """Encode the initial read AnnData object. Categorical values could be either passed via parameters or autodetected.
    The categorical values are also stored in obs and uns (for keeping the original, unencoded values).
    The current encoding modes for each variable are also stored in uns (`current_encodings` key).
    Variable names in var are updated according to the encoding modes used.
    A variable name starting with `ehrapycat_` indicates an encoded column (or part of it).

    Available encodings are:
        1. one-hot encoding (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
        2. label encoding (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
        3. count encoding (https://contrib.scikit-learn.org/category_encoders/count.html)

    Args:
        ann_data: The inital AnnData object
        autodetect: Autodetection of categorical values
        encodings: Only needed if autodetect set to False.
        A dict containing the categorical name and the encoding mode for the respective column.

    Returns:
        An :class:`~anndata.AnnData` object with the encoded values in X

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.io.read(...)
            # encode col1 and col2 using label encoding and encode col3 using one hot encoding
            ep.encode.encode(adata, autodetect=False, {'label_encoding': ['col1', 'col2'], 'one_hot_encoding': ['col3']})
    """
    return Encoder.encode(ann_data, autodetect, encodings)


def undo_encoding(
    adata: AnnData, columns: str = "all", from_cache_file: bool = False, cache_file: str = None
) -> AnnData:
    """Undo the current encodings applied to some or all columns in X. This currently resets the AnnData object to its initial state.
    Args:
        adata: The AnnData object
        columns: The names of the columns to reset encoding for. Defaults to all columns.
        from_cache_file: Whether to reset all encodings by reading from a cached .h5ad file, if available. This resets the AnnData object to its initial
        state.
        TODO replace this once settings.cache_dir is available
        cache_file: The filename of the cache file to read from

    Returns:
        A (partially) encoding reset AnnData object
    """
    return Encoder.undo_encoding(adata, columns, from_cache_file, cache_file)
