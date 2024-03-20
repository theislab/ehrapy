from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from scanpy import AnnData
from tableone import TableOne

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure


def _check_columns_exist(df, columns) -> None:
    missing_columns = set(columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns {list(missing_columns)} not found in DataFrame.")


def _check_no_new_categories(df: pd.DataFrame, categorical: pd.DataFrame, categorical_labels: dict) -> None:
    """Check if new categories have been added to the categorical columns: this would break the plotting logic."""
    for col in categorical:
        categories_present = df[col].astype("category").cat.categories
        categories_expected = categorical_labels[col]
        diff = set(categories_present) - set(categories_expected)
        if diff:
            raise ValueError(f"New category in {col}: {diff}")


def _detect_categorical_columns(data) -> list:
    # TODO grab this from ehrapy once https://github.com/theislab/ehrapy/issues/662 addressed
    numeric_cols = set(data.select_dtypes("number").columns)
    categorical_cols = set(data.columns) - numeric_cols

    return list(categorical_cols)


class CohortTracker:
    """Track cohort changes over multiple filtering or processing steps.

    This class offers functionality to track and plot cohort changes over multiple filtering or processing steps,
    enabling the user to monitor the impact of each step on the cohort.

    Tightly interacting with the `tableone` package [1].

    Args:
        adata: AnnData object to track.
        columns: Columns to track. If `None`, all columns will be tracked. Defaults to `None`.
        categorical: Columns that contain categorical variables, if None will be inferred from the data. Defaults to `None`.

    References:
        [1] Tom Pollard, Alistair E.W. Johnson, Jesse D. Raffa, Roger G. Mark; tableone: An open source Python package for producing summary statistics for research papers, Journal of the American Medical Informatics Association, Volume 24, Issue 2, 1 March 2017, Pages 267–271, https://doi.org/10.1093/jamia/ocw117
    """

    def __init__(self, adata: AnnData, columns: Sequence = None, categorical: Sequence = None) -> None:
        if not isinstance(adata, AnnData):
            raise ValueError("adata must be an AnnData.")

        self.columns = columns if columns is not None else list(adata.obs.columns)

        if columns is not None:
            _check_columns_exist(adata.obs, columns)
        if categorical is not None:
            _check_columns_exist(adata.obs, categorical)
            if set(categorical).difference(set(self.columns)):
                raise ValueError("categorical columns must be in the (selected) columns.")

        self._tracked_steps: int = 0
        self._tracked_text: list = []
        self._tracked_operations: list = []

        # if categorical columns specified, use them
        # else, follow tableone's logic
        self.categorical = (
            categorical if categorical is not None else _detect_categorical_columns(adata.obs[self.columns])
        )

        self._categorical_categories: dict = {
            col: adata.obs[col].astype("category").cat.categories for col in self.categorical
        }
        self._tracked_tables: list = []

    def __call__(self, adata: AnnData, label: str = None, operations_done: str = None, **tableone_kwargs: dict) -> None:
        if not isinstance(adata, AnnData):
            raise ValueError("adata must be an AnnData.")

        _check_columns_exist(adata.obs, self.columns)
        _check_no_new_categories(adata.obs, self.categorical, self._categorical_categories)

        # track a small text with each tracking step, for the flowchart
        track_text = label if label is not None else f"Cohort {self.tracked_steps}"
        track_text += "\n (n=" + str(adata.n_obs) + ")"
        self._tracked_text.append(track_text)

        # track a small text with the operations done
        self._tracked_operations.append(operations_done)
        self._tracked_steps += 1

        # track new tableone object
        t1 = TableOne(adata.obs, columns=self.columns, categorical=self.categorical, **tableone_kwargs)
        self._tracked_tables.append(t1)

    def _get_cat_data(self, table_one: TableOne, col: str) -> pd.DataFrame:
        cat_pct: dict = {category: [] for category in self._categorical_categories[col]}

        for cat in self._categorical_categories[col]:
            # this checks if all instances of category "cat" which were initially present have been
            # lost (e.g. due to filtering steps): in this case, this category
            # will be assigned a percentage of 0
            # for categorized columns (e.g. gender 1.0/0.0), str(cat) helps to avoid considering the category as a float
            if str(cat) in table_one.cat_table["Overall"].loc[col].index:
                pct = float(table_one.cat_table["Overall"].loc[(col, str(cat))].split("(")[1].split(")")[0])
            else:
                pct = 0

            cat_pct[cat] = [pct]
        return pd.DataFrame(cat_pct).T[0]

    def _get_num_data(self, table_one: TableOne, col: str):
        return table_one.cont_table["Overall"].loc[(col, "")]

    def _check_legend_labels(self, legend_handels: dict) -> None:
        if not isinstance(legend_handels, dict):
            raise ValueError("legend_labels must be a dictionary.")

        values = [
            category for column_categories in self._categorical_categories.values() for category in column_categories
        ]

        missing_keys = [key for key in legend_handels if key not in values and key not in self.columns]

        if missing_keys:
            raise ValueError(f"legend_labels key(s) {missing_keys} not found as categories or numerical column names.")

    def _check_yticks_labels(self, yticks_labels: dict) -> None:
        if not isinstance(yticks_labels, dict):
            raise ValueError("yticks_labels must be a dictionary.")

        # Find keys in legend_handels that are not in values or self.columns
        missing_keys = [key for key in yticks_labels if key not in self.columns]

        if missing_keys:
            raise ValueError(f"legend_handels key(s) {missing_keys} not found as categories or numerical column names.")

    @property
    def tracked_steps(self):
        """Number of tracked steps."""
        return self._tracked_steps

    @property
    def tracked_tables(self):
        """List of :class:`~tableone.TableOne` objects of each logging step."""
        return self._tracked_tables

    def plot_cohort_barplot(
        self,
        subfigure_title: bool = False,
        color_palette: str = "colorblind",
        yticks_labels: dict = None,
        legend_labels: dict = None,
        show: bool = True,
        ax: Axes | Sequence[Axes] = None,
        subplots_kwargs: dict = None,
        legend_kwargs: dict = None,
    ) -> None | list[Axes] | tuple[Figure, list[Axes]]:
        """Plot the cohort change over the tracked steps.

        Create stacked bar plots to monitor cohort changes over the steps tracked with `CohortTracker`.

        Args:
            subfigure_title: If `True`, each subplot will have a title with the `label` provided during tracking.
            color_palette: The color palette to use for the plot. Default is "colorblind".
            yticks_labels: Dictionary to rename the axis labels. If `None`, the original labels will be used. The keys should be the column names.
            legend_labels: Dictionary to rename the legend labels. If `None`, the original labels will be used. For categoricals, the keys should be the categories. For numericals, the key should be the column name.
            show: If `True`, the plot will be shown. If `False`, plotting handels are returned.
            ax: If `None`, a new figure and axes will be created. If an axes object is provided, the plot will be added to it.
            subplots_kwargs: Additional keyword arguments for the subplots.
            legend_kwargs: Additional keyword arguments for the legend.

        Returns:
            If `show=True`, returns `None`. Else, if no ax is passed, returns a tuple  (:class:`~matplotlib.figure.Figure`, :class:`~list`(:class:`~matplotlib.axes.Axes`), else a :class:`~list`(:class:`~matplotlib.axes.Axes`).

        Examples:
                >>> import ehrapy as ep
                >>> adata = ep.dt.diabetes_130_fairlearn(
                >>>     columns_obs_only=["gender", "race", "num_procedures"]
                >>> )
                >>> cohort_tracker = ep.tl.CohortTracker(adata, categorical=["gender", "race"])
                >>> cohort_tracker(adata, "Initial Cohort")
                >>> adata = adata[:1000]
                >>> cohort_tracker(adata, "Filtered Cohort")
                >>> cohort_tracker.plot_cohort_barplot(
                >>>     subfigure_title=True,
                >>>     color_palette="tab20",
                >>>     yticks_labels={
                >>>         "race": "Race [%]",
                >>>         "gender": "Gender [%]",
                >>>         "num_procedures": "#Procedures [mean (stdev)]",
                >>>     },
                >>>     legend_labels={
                >>>         "Unknown/Invalid": "Unknown",
                >>>         "num_procedures": "#Procedures",
                >>>     },
                >>>     legend_kwargs={"bbox_to_anchor": (1, 1.4)},
                >>> )

            .. image:: /_static/docstring_previews/cohort_tracking.png
        """
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs

        legend_labels = {} if legend_labels is None else legend_labels
        self._check_legend_labels(legend_labels)

        yticks_labels = {} if yticks_labels is None else yticks_labels
        self._check_yticks_labels(yticks_labels)

        if ax is None:
            fig, axes = plt.subplots(self.tracked_steps, 1, **subplots_kwargs)
        else:
            axes = ax

        legend_handles = []

        # if only one step is tracked, axes object is not iterable
        if isinstance(axes, Axes):
            axes = [axes]

        # need to get the number of required colors first
        num_colors = 0
        for _, col in enumerate(self.columns):
            if col in self.categorical:
                num_colors += len(self._categorical_categories[col])
            else:
                num_colors += 1

        colors = sns.color_palette(color_palette, num_colors)

        # each tracked step is a subplot
        for idx, single_ax in enumerate(axes):
            # this is needed to avoid the x-axis overlapping the bars, which else would pop up sometimes in notebooks
            single_ax.grid(False)

            if subfigure_title:
                single_ax.set_title(self._tracked_text[idx])

            color_count = 0
            # iterate over the tracked columns in the dataframe
            for pos, col in enumerate(self.columns):
                if col in self.categorical:
                    data = self._get_cat_data(self.tracked_tables[idx], col)
                else:
                    data = [self._get_num_data(self.tracked_tables[idx], col)]

                cumwidth = 0
                # for categoricals, plot multiple bars
                if col in self.categorical:
                    col_legend_handles = []
                    for i, value in enumerate(data):
                        stacked_bar_color = colors[color_count]
                        color_count += 1
                        single_ax.barh(
                            pos,
                            value,
                            left=cumwidth,
                            color=stacked_bar_color,
                            height=0.7,
                            edgecolor="black",
                            linewidth=0.6,
                        )

                        if value > 5:
                            # Add proportion numbers to the bars
                            width = value
                            single_ax.text(
                                cumwidth + width / 2,
                                pos,
                                f"{value:.1f}",
                                ha="center",
                                va="center",
                                color="white",
                                fontweight="bold",
                            )

                        single_ax.set_yticks([])
                        single_ax.set_xticks([])
                        cumwidth += value

                        if idx == 0:
                            name = (
                                legend_labels[data.index[i]] if data.index[i] in legend_labels.keys() else data.index[i]
                            )

                            col_legend_handles.append(Patch(color=stacked_bar_color, label=name))
                    legend_handles.append(col_legend_handles)

                # for numericals, plot a single bar
                else:
                    stacked_bar_color = colors[color_count]
                    color_count += 1
                    single_ax.barh(
                        pos,
                        100,
                        left=cumwidth,
                        color=stacked_bar_color,
                        height=0.8,
                        edgecolor="black",
                        linewidth=0.6,
                    )
                    single_ax.text(
                        100 / 2,
                        pos,
                        data[0],
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                    )
                    if idx == 0:
                        name = legend_labels[col] if col in legend_labels.keys() else col

                        legend_handles.append([Patch(color=stacked_bar_color, label=name)])

            single_ax.set_yticks(range(len(self.columns)))
            names = [
                yticks_labels[col] if yticks_labels is not None and col in yticks_labels.keys() else col
                for col in self.columns
            ]
            single_ax.set_yticklabels(names)

        # These list of lists is needed to reverse the order of the legend labels,
        # making the plot much more readable
        legend_handles.reverse()
        legend_handels = [item for sublist in legend_handles for item in sublist]

        tot_legend_kwargs = {"loc": "best", "bbox_to_anchor": (1, 1)}
        if legend_kwargs is not None:
            tot_legend_kwargs.update(legend_kwargs)

        plt.legend(handles=legend_handels, **tot_legend_kwargs)

        if show:
            plt.tight_layout()
            plt.show()
            return None

        # to be able to e.g. save the figure, the fig object helps a lot
        # if users have passed an ax, they likely have the one belonging to ax.
        # else, give the one belonging to the created axes
        else:
            if ax is None:
                return fig, axes
            else:
                return axes

    def plot_flowchart(
        self,
        title: str = None,
        arrow_size: float = 0.7,
        show: bool = True,
        ax: Axes = None,
        bbox_kwargs: dict = None,
        arrowprops_kwargs: dict = None,
    ) -> None | list[Axes] | tuple[Figure, list[Axes]]:
        """Flowchart over the tracked steps.

        Create a simple flowchart of data preparation steps tracked with `CohortTracker`.

        Args:
            arrow_size: The size of the arrows in the plot. Default is 0.7.
            show: If `True`, the plot will be displayed. If `False`, plotting handels are returned.
            ax: If `None`, a new figure and axes will be created. If an axes object is provided, the plot will be added to it.
            bbox_kwargs: Additional keyword arguments for the node boxes.
            arrowprops_kwargs: Additional keyword arguments for the arrows.

        Returns:
            If `show=True`, returns `None`. Else, if no ax is passed, returns a tuple  (:class:`~matplotlib.figure.Figure`, :class:`~list`(:class:`~matplotlib.axes.Axes`), else a :class:`~list`(:class:`~matplotlib.axes.Axes`).

        Examples:
                >>> import ehrapy as ep
                >>> adata = ep.dt.diabetes_130_fairlearn(columns_obs_only="gender", "race")
                >>> cohort_tracker = ep.tl.CohortTracker(adata)
                >>> cohort_tracker(adata, label="Initial Cohort")
                >>> adata = adata[:1000]
                >>> cohort_tracker(adata, label="Reduced Cohort", operations_done="filtered to first 1000 entries")
                >>> adata = adata[:500]
                >>> cohort_tracker(
                ...     adata,
                ...     label="Further reduced Cohort",
                ...     operations_done="filtered to first 500 entries",
                ... )
                >>> cohort_tracker.plot_flowchart(title="Flowchart of Data Processing", show=True)

            .. image:: /_static/docstring_previews/flowchart.png
        """
        if ax is None:
            fig, axes = plt.subplots()
        else:
            axes = ax
        axes.set_aspect("equal")

        if title is not None:
            axes.set_title(title)
        # Define positions for the nodes
        # heuristic to avoid oversized gaps
        max_pos = min(0.3 * self.tracked_steps, 1)
        y_positions = np.linspace(max_pos, 0, self.tracked_steps)

        node_labels = self._tracked_text

        tot_bbox_kwargs = {"boxstyle": "round,pad=0.3", "fc": "lightblue", "alpha": 0.5}
        if bbox_kwargs is not None:
            tot_bbox_kwargs.update(bbox_kwargs)
        for _, (y, label) in enumerate(zip(y_positions, node_labels)):
            axes.annotate(
                label,
                xy=(0, y),
                xytext=(0, y),
                ha="center",
                va="center",
                bbox=tot_bbox_kwargs,
            )

        for i in range(len(self._tracked_operations) - 1):
            axes.annotate(
                self._tracked_operations[i + 1],
                xy=(0, (y_positions[i] + y_positions[i + 1]) / 2),
                xytext=(0.01, (y_positions[i] + y_positions[i + 1]) / 2),
            )

        tot_arrowprops_kwargs = {"arrowstyle": "->", "connectionstyle": "arc3", "color": "gray"}
        if arrowprops_kwargs is not None:
            tot_arrowprops_kwargs.update(arrowprops_kwargs)
        for i in range(len(self._tracked_operations) - 1):
            arrow_length = (
                y_positions[i] - y_positions[i + 1] - (y_positions[i] - y_positions[i + 1]) * (1 - arrow_size)
            )
            axes.annotate(
                "",
                xy=(0, (y_positions[i] + y_positions[i + 1]) / 2 - arrow_length / 2),
                xytext=(0, (y_positions[i] + y_positions[i + 1]) / 2 + arrow_length / 2),
                arrowprops=tot_arrowprops_kwargs,
            )

        # required to center the plot
        axes.set_xlim(-0.5, 0.5)
        axes.set_ylim(0, 1.1)

        axes.set_axis_off()
        if show:
            plt.show()
            return None
        else:
            if ax is None:
                return fig, axes
            else:
                return axes
