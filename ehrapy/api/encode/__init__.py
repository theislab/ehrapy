from typing import Dict, List

from anndata import AnnData

from ehrapy.api.encode.encode import Encoder


def encode(
    ann_data: AnnData, autodetect: bool = False, categoricals_encode_mode: Dict[str, List[str]] = None
) -> AnnData:
    """Encode the initial read AnnData object. Categorical values could be either passed via parameters or autodetected.
    The categorical values are also stored in obs and uns (for keeping the original, unencoded values). The current encoding modes for each variable
    are also stored in uns (`current_encodings` key). Variable names in var are updated according to the encoding modes used. A variable name
    starting with `ehrapycat_` indicates an encoded column (or part of it).

    Available encodings are:

        1. one-hot encoding (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
        2. label encoding (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
        3. count encoding (https://contrib.scikit-learn.org/category_encoders/count.html)

    Parameters:
        ann_data
            The inital AnnData object
        autodetect
            Autodetection of categorical values
        categoricals_encode_mode
            Only needed if autodetect set to False. A dict containing the categorical name
            and the encoding mode for the respective column. Below an example:

            .. code-block:: python

                import ehrapy.api as ehp
                adata = ehp.io.read(...)
                # encode col1 and col2 using label encoding and encode col3 using one hot encoding
                ehp.encode.encode(adata, autodetect=False, {'label_encoding': ['col1', 'col2'], 'one_hot_encoding': ['col3']})

    Returns:
            An :class:`~anndata.AnnData` object with the encoded values in X
    """
    return Encoder.encode(ann_data, autodetect, categoricals_encode_mode)
