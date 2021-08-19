from anndata import AnnData
from rich import print
from rich.text import Text
from rich.tree import Tree

from ehrapy.api.encode.encode import Encoder


def adata_type_overview(adata: AnnData, sorted: bool = False) -> None:
    """Prints the current state of an AnnData object in a tree format.

    Args:
        adata: AnnData object to examine
        sorted: Whether the tree output should be sorted
    """
    # TODO implement sorted!
    encoding_mapping = {
        encoding: encoding.replace("encoding", "").replace("_", " ").strip() for encoding in Encoder.available_encodings
    }

    tree = Tree(
        f"[bold green]Variable names for AnnData object with {len(adata.var_names)} vars",
        guide_style="underline2 bright_blue",
    )
    if list(adata.obs.columns):
        branch = tree.add("Obs", style="bold green")
        for categorical in list(adata.obs.columns):
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

    for other_vars in list(adata.var_names.values):
        if not other_vars.startswith("ehrapycat"):
            branch_num.add(f"{other_vars}", style="blue")

    print(tree)
