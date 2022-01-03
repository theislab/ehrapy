from __future__ import annotations

import pandas as pd
from anndata import AnnData
from mudata import MuData
from rich import print
from rich.text import Text
from rich.tree import Tree

from ehrapy.api.preprocessing.encoding._encode import available_encodings


def type_overview(data: MuData | AnnData, sort: bool = False, sort_reversed: bool = False) -> None:
    """Prints the current state of an :class:`~anndata.AnnData` or :class:`~mudata.MuData` object in a tree format.

    Args:
        data: :class:`~anndata.AnnData` or :class:`~mudata.MuData` object to display
        sort: Whether the tree output should be in sorted order
        sort_reversed: Whether to sort in reversed order or not

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.dt.mimic_2(encode=True)
            ep.pp.type_overview(adata)
    """
    if isinstance(data, AnnData):
        _adata_type_overview(data, sort, sort_reversed)
    elif isinstance(data, MuData):
        _mudata_type_overview(data, sort, sort_reversed)
    else:
        print(f"[b red]Unable to present object of type {type(data)}. Can only display AnnData or MuData objects!")
        raise EhrapyRepresentationError


def _adata_type_overview(adata: AnnData, sort: bool = False, sort_reversed: bool = False) -> None:
    """Display the :class:`~anndata.AnnData object in its current state (encoded and unencoded variables, obs)

    Args:
        adata: The :class:`~anndata.AnnData object to display
        sort: Whether to sort output or not
        sort_reversed: Whether to sort output in reversed order or not
    """
    encoding_mapping = {
        encoding: encoding.replace("encoding", "").replace("_", " ").strip() for encoding in available_encodings
    }

    tree = Tree(
        f"[b green]Variable names for AnnData object with {len(adata.var_names)} vars and {len(adata.obs_names)} obs",
        guide_style="underline2 bright_blue",
    )
    is_encoded = False
    if "current_encodings" in adata.uns.keys():
        is_encoded = True
        original_values = adata.uns["original_values_categoricals"]
        branch = tree.add("🔐 Encoded variables", style="b green")
        encoded_list = sorted(original_values.keys(), reverse=sort_reversed) if sort else list(original_values.keys())
        for categorical in encoded_list:
            unique_categoricals = pd.unique(original_values[categorical].flatten())
            categorical_type = pd.api.types.infer_dtype(unique_categoricals)
            is_nan = pd.DataFrame(unique_categoricals).isnull().values.any()
            branch.add(
                f"[blue]{categorical} -> {len(unique_categoricals) - 1 if is_nan else len(unique_categoricals)} categories;"
                f" [green]{encoding_mapping[adata.uns['current_encodings'][categorical]]} [blue]encoded; [green]original data type: [blue]{categorical_type}"
            )

    branch_num = tree.add(Text("🔓 Unencoded variables"), style="b green")

    var_names = sorted(list(adata.var_names.values), reverse=sort_reversed) if sort else list(adata.var_names.values)

    for other_vars in var_names:
        idx = list(adata.var_names.values).index(other_vars)
        if is_encoded:
            data_type = "numerical"
        else:
            unique_categoricals = pd.unique(adata.X[:, idx : idx + 1].flatten())
            data_type = pd.api.types.infer_dtype(unique_categoricals)
        if not other_vars.startswith("ehrapycat"):
            branch_num.add(f"[blue]{other_vars} -> [green]data type: [blue]{data_type}")

    if sort:
        print(
            "[b yellow]Displaying AnnData object in sorted mode. "
            "Note that this might not be the exact same order of the variables in X or var are stored!"
        )
    print(tree)


def _mudata_type_overview(mudata: MuData, sort: bool = False, sort_reversed: bool = False) -> None:
    """Display the :class:`~mudata.MuData object in its current state (:class:`~anndata.AnnData objects with obs, shapes)

    Args:
        mudata: The :class:`~mudata.MuData object to display
        sort: Whether to sort output or not
        sort_reversed: Whether to sort output in reversed order or not
    """
    tree = Tree(
        f"[b green]Variable names for AnnData object with {len(mudata.var_names)} vars, {len(mudata.obs_names)} obs and {len(mudata.mod.keys())} modalities\n",
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


def _single_quote_string(name: str) -> str:
    """Single quote a string to inject it into f-strings, since backslashes cannot be in double f-strings."""
    return f"'{name}'"


class EhrapyRepresentationError(ValueError):
    pass
