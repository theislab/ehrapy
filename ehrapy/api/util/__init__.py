from anndata import AnnData
from rich import print
from rich.text import Text
from rich.tree import Tree


def vars_tree(ann_data: AnnData) -> None:
    """Prints a tree of all vars of an AnnData object.

    Args:
        ann_data: The AnnData object to print the vars from
    """
    tree = Tree(
        f"[bold green]Variable names for AnnData object with {len(ann_data.raw.var_names)} variables",
        guide_style="underline2 bright_blue",
    )
    # TODO generalize this method
    branch = tree.add(Text("📄 " + "Categoricals"), style="bold green")
    branch.add("Day_ICU_intime with 7 different categories", style="blue")
    branch.add("Service_unit with 3 different categories", style="blue")

    branch_num = tree.add(Text("❶ " + "Numericals"), style="bold green")

    for i in range(2, len(ann_data.raw.var_names)):
        branch_num.add(ann_data.raw.var_names[i], style="blue")

    print(tree)
