from anndata import AnnData
from rich import print
from rich.text import Text
from rich.tree import Tree


def ann_data_tree(adata: AnnData) -> None:
    """Prints the current state of an AnnData object in a tree format.

    Parameter:
        adata
            The AnnData object
    """
    encoding_mapper = {"label_encoding": "label", "one_hot_encoding": "one hot", "count_encoding": "count"}

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
                            f"{categorical} 🔐; {len(adata.obs[categorical].unique())} different categories; currently {encoding_mapper[adata.uns['current_encodings'][categorical]]} encoded"
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
