from typing import Dict, List

from anndata import AnnData
from rich import print
from rich.text import Text
from rich.tree import Tree

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
    """Undo the current encodings applied to all columns in X. This currently resets the AnnData object to its initial state.
    Args:
        adata: The AnnData object
        columns: The names of the columns to reset encoding for. Defaults to all columns.
        from_cache_file: Whether to reset all encodings by reading from a cached .h5ad file, if available.
        This resets the AnnData object to its initial state.
        TODO replace this once settings.cache_dir is available
        cache_file: The filename of the cache file to read from

    Returns:
        A (partially) encoding reset AnnData object

    Example:
       .. code-block:: python

           import ehrapy.api as ep
           # adata_encoded is a encoded AnnData object
           adata_undone = ep.encode.undo_encoding(adata_encoded)
           # adata_undone is a fully reset AnnData object with no encodings
    """
    return Encoder.undo_encoding(adata, columns, from_cache_file, cache_file)


def type_overview(adata: AnnData, sort: bool = False, sort_reversed: bool = False) -> None:
    """Prints the current state of an AnnData object in a tree format.

    Args:
        adata: AnnData object to examine
        sort: Whether the tree output should be sorted
        sort_reversed: Whether to sort in reversed order or not
    """
    encoding_mapping = {
        encoding: encoding.replace("encoding", "").replace("_", " ").strip() for encoding in Encoder.available_encodings
    }

    tree = Tree(
        f"[bold green]Variable names for AnnData object with {len(adata.var_names)} vars",
        guide_style="underline2 bright_blue",
    )
    if list(adata.obs.columns):
        branch = tree.add("Obs", style="bold green")
        column_list = sorted(adata.obs.columns, reverse=sort_reversed) if sort else list(adata.obs.columns)
        for categorical in column_list:
            if "current_encodings" in adata.uns.keys():
                if categorical in adata.uns["current_encodings"].keys():
                    branch.add(
                        Text(
                            f"{categorical} 🔐; {len(adata.obs[categorical].unique())} different categories;"
                            f" currently {encoding_mapping[adata.uns['current_encodings'][categorical]]} encoded"
                        ),
                        style="blue",
                    )
                else:
                    branch.add(Text(f"{categorical}; moved from X to obs"), style="blue")
            else:
                branch.add(Text(f"{categorical}; moved from X to obs"), style="blue")

    branch_num = tree.add(Text("🔓 Unencoded variables"), style="bold green")

    var_names = sorted(list(adata.var_names.values), reverse=sort_reversed) if sort else list(adata.var_names.values)
    for other_vars in var_names:
        if not other_vars.startswith("ehrapycat"):
            branch_num.add(f"{other_vars}", style="blue")

    if sort:
        print(
            "[bold yellow]Displaying AnnData object in sorted mode. "
            "Note that this might not be the exact same order of the variables in X or var are stored!"
        )
    print(tree)
