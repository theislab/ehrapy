from anndata import AnnData
from rich import print
from rich.text import Text
from rich.tree import Tree


def vars_tree(ann_data: AnnData) -> None:
    """Prints a tree of all variables of an AnnData object.

    Args:
        ann_data: The AnnData object to print the variables from
    """
    encoding_mapper = {"label_encoding": "label", "one_hot_encoding": "one hot", "count_encoding": "count"}

    tree = Tree(
        f"[bold green]Variable names for AnnData object with {len(ann_data.var_names)} variables",
        guide_style="underline2 bright_blue",
    )
    if list(ann_data.obs.columns):
        branch = tree.add("Obs", style="bold green")
        for categorical in list(ann_data.obs.columns):
            if "current_encodings" in ann_data.uns.keys():
                if categorical in ann_data.uns["current_encodings"].keys():
                    branch.add(
                        Text(
                            f"{categorical} 🔐; {len(ann_data.obs[categorical].unique())} different categories; currently {encoding_mapper[ann_data.uns['current_encodings'][categorical]]} encoded"
                        ),
                        style="blue",
                    )
                else:
                    branch.add(Text(f"{categorical}; moved from X to obs"), style="blue")
            else:
                branch.add(Text(f"{categorical}; moved from X to obs"), style="blue")

    branch_num = tree.add(Text("🔓 Unencoded variables"), style="bold green")

    for other_vars in list(ann_data.var_names.values):
        if not other_vars.startswith("ehrapycat"):
            branch_num.add(f"{other_vars}", style="blue")

    print(tree)
