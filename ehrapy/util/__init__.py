from rich import print
from rich.text import Text
from rich.tree import Tree


def view_vars(ann_data):
    tree = Tree(
        f"[bold red]Variable names for AnnData object with {len(ann_data.raw.var_names)} variables",
        guide_style="underline2 bright_blue",
    )
    branch = tree.add(Text("📄 " + "Categoricals"), style="bold green")
    branch.add("Day_ICU_intime with 7 different categories", style="blue")
    branch.add("Service_unit with 3 different categories", style="blue")

    branch_num = tree.add(Text("❶ " + "Numericals"), style="bold green")

    for i in range(2, len(ann_data.raw.var_names)):
        branch_num.add(ann_data.raw.var_names[i], style="blue")

    print(tree)
