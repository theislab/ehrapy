from anndata import AnnData
from rich import print
from rich.text import Text
from rich.tree import Tree

from ehrapy.api.encode.encode import Encoder


def adata_type_overview(adata: AnnData, sort: bool = False, sort_reversed: bool = False) -> None:
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
