from __future__ import annotations

from anndata import AnnData
from rich.console import Console
from rich.table import Table


def qc_metrics(adata: AnnData, extra_columns: list[str] | None = None) -> None:  # pragma: no cover
    """Plots the calculated quality control metrics for var of adata. Per default this will display the following features:
    ``missing_values_abs``, ``missing_values_pct``, ``mean``, ``median``, ``standard_deviation``, ``max``, ``min``.

    Args:
        adata: Annotated data matrix.
        extra_columns: List of custom (qc) var columns to be displayed additionally.

    """
    table = Table(title="[bold blue]Ehrapy qc metrics of var")
    # add special column header for the column name
    table.add_column("[bold blue]Column name", justify="right", style="bold green")
    var_names = list(adata.var_names)
    # default qc columns added to var
    fixed_qc_columns = [
        "missing_values_abs",
        "missing_values_pct",
        "mean",
        "median",
        "standard_deviation",
        "min",
        "max",
    ]
    # update columns to display with extra columns (if any)
    columns_to_display = fixed_qc_columns if not extra_columns else fixed_qc_columns + extra_columns
    # check whether all columns exist (qc has been executed before and extra columns are var columns)
    if (set(columns_to_display) & set(adata.var.columns)) != set(columns_to_display):
        raise QCDisplayError(
            "Cannot display QC metrics of current AnnData object. Either QC has not been executed before or "
            "some column(s) of the extra_columns parameter are not in var!"
        )
    vars_to_display = adata.var[columns_to_display]
    # add column headers
    for col in vars_to_display:
        table.add_column(f"[bold blue]{col}", justify="right", style="bold green")
    for var in range(len(vars_to_display)):
        table.add_row(var_names[var], *map(str, list(vars_to_display.iloc[var])))

    console = Console()
    console.print(table)


class QCDisplayError(Exception):
    pass
