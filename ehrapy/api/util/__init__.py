import sys
from typing import Union

from anndata import AnnData
from mudata import MuData
from rich import print
from rich.text import Text
from rich.tree import Tree

from ehrapy.api.encode.encode import Encoder


def data_type_overview(data: Union[MuData, AnnData], sort: bool = False, sort_reversed: bool = False) -> None:
    """Prints the current state of an AnnData or MuData object in a tree format.

    Args:
        data: AnnData or MuData object to display
        sort: Whether the tree output should be in sorted order
        sort_reversed: Whether to sort in reversed order or not
    """
    if isinstance(data, AnnData):
        _adata_type_overview(data, sort, sort_reversed)
    elif isinstance(data, MuData):
        _mudata_type_overview(data, sort, sort_reversed)
    else:
        print(f"[b red]Unable to present object of type {type(data)}. Can only display AnnData or Mudata objects!")
        sys.exit(1)


def _adata_type_overview(adata: AnnData, sort: bool = False, sort_reversed: bool = False) -> None:
    encoding_mapping = {
        encoding: encoding.replace("encoding", "").replace("_", " ").strip() for encoding in Encoder.available_encodings
    }

    tree = Tree(
        f"[b green]Variable names for AnnData object with {len(adata.var_names)} vars and {len(adata.obs_names)} obs",
        guide_style="underline2 bright_blue",
    )
    if list(adata.obs.columns):
        branch = tree.add("Obs", style="b green")
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

    branch_num = tree.add(Text("🔓 Unencoded variables"), style="b green")

    var_names = sorted(list(adata.var_names.values), reverse=sort_reversed) if sort else list(adata.var_names.values)
    for other_vars in var_names:
        idx = 0
        if not other_vars.startswith("ehrapycat"):
            branch_num.add(f"{other_vars}", style="blue")
        idx += 1

    if sort:
        print(
            "[b yellow]Displaying AnnData object in sorted mode. "
            "Note that this might not be the exact same order of the variables in X or var are stored!"
        )
    print(tree)


def _mudata_type_overview(mudata: AnnData, sort: bool = False, sort_reversed: bool = False) -> None:

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
