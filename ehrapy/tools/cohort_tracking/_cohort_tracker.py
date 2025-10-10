from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from ehrdata.core.constants import CATEGORICAL_TAG
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tableone import TableOne

from ehrapy._compat import use_ehrdata
from ehrapy.anndata._feature_specifications import _detect_feature_type

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData
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


import matplotlib.text as mtext


class LegendTitle:
    def __init__(self, text_props=None):
        self.text_props = text_props or {}

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, **self.text_props)
        handlebox.add_artist(title)
        return title


class CohortTracker:
    """Track cohort changes over multiple filtering or processing steps.

    This class offers functionality to track and plot cohort changes over multiple filtering or processing steps,
    enabling the user to monitor the impact of each step on the cohort.

    Tightly interacting with the `tableone` package [1].

    Args:
        edata: Central data object.
        columns: Columns to track. If `None`, all columns will be tracked.
        categorical: Columns that contain categorical variables, if None will be inferred from the data.

    References:
        [1] Tom Pollard, Alistair E.W. Johnson, Jesse D. Raffa, Roger G. Mark;
        tableone: An open source Python package for producing summary statistics for research papers, Journal of the American Medical Informatics Association, Volume 24, Issue 2, 1 March 2017, Pages 267–271, https://doi.org/10.1093/jamia/ocw117
    """

    @use_ehrdata(deprecated_after="1.0.0")
    def __init__(
        self,
        edata: AnnData | EHRData,
        columns: Sequence | None = None,
        categorical: Sequence | None = None,
    ) -> None:
        if not isinstance(edata, AnnData):
            raise ValueError("edata must be an AnnData or EHRData.")

        self.columns = columns if columns is not None else list(edata.obs.columns)

        if columns is not None:
            _check_columns_exist(edata.obs, columns)
        if categorical is not None:
            _check_columns_exist(edata.obs, categorical)
            if set(categorical).difference(set(self.columns)):
                raise ValueError("categorical columns must be in the (selected) columns.")

        self._tracked_steps: int = 0
        self._tracked_text: list = []
        self._tracked_operations: list = []

        # if categorical columns specified, use them, else infer the feature types
        self.categorical = (
            categorical
            if categorical is not None
            else [
                col
                for col in edata.obs[self.columns].columns
                if _detect_feature_type(edata.obs[col])[0] == CATEGORICAL_TAG
            ]
        )

        self._categorical_categories: dict = {
            col: edata.obs[col].astype("category").cat.categories for col in self.categorical
        }
        self._tracked_tables: list = []

    @use_ehrdata(deprecated_after="1.0.0")
    def __call__(
        self, edata: AnnData | EHRData, label: str = None, operations_done: str = None, **tableone_kwargs: dict
    ) -> None:
        if not isinstance(edata, AnnData):
            raise ValueError("edata must be an AnnData or EHRData.")

        _check_columns_exist(edata.obs, self.columns)
        _check_no_new_categories(edata.obs, self.categorical, self._categorical_categories)

        # track a small text with each tracking step, for the flowchart
        track_text = label if label is not None else f"Cohort {self.tracked_steps}"
        track_text += "\n (n=" + str(edata.n_obs) + ")"
        self._tracked_text.append(track_text)

        # track a small text with the operations done
        self._tracked_operations.append(operations_done)
        self._tracked_steps += 1

        # track new tableone object
        t1 = TableOne(edata.obs, columns=self.columns, categorical=self.categorical, **tableone_kwargs)
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

    def _check_legend_subtitle_names(self, legend_subtitles_names: dict) -> None:
        if not isinstance(legend_subtitles_names, dict):
            raise ValueError("legend_subtitles_names must be a dictionary.")

        # Find keys in legend_handels that are not in values or self.columns
        missing_keys = [key for key in legend_subtitles_names if key not in self.columns]

        if missing_keys:
            raise ValueError(
                f"legend_subtitles_names key(s) {missing_keys} not found as categories or numerical column names."
            )

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
        *,
        subfigure_title: bool = False,
        color_palette: str = "colorblind",
        yticks_labels: dict = None,
        legend_labels: dict = None,
        legend_subtitles: bool = True,
        legend_subtitles_names: dict = None,
        show: bool = True,
        ax: Axes | Sequence[Axes] = None,
        fontsize: int = 10,
        subplots_kwargs: dict = None,
        legend_kwargs: dict = None,
    ) -> None | list[Axes] | tuple[Figure, list[Axes]]:
        """Plot the cohort change over the tracked steps.

        Create stacked bar plots to monitor cohort changes over the steps tracked with `CohortTracker`.

        Args:
            subfigure_title: If `True`, each subplot will have a title with the `label` provided during tracking.
            color_palette: The color palette to use for the plot.
            yticks_labels: Dictionary to rename the axis labels. If `None`, the original labels will be used.
                The keys should be the column names.
            legend_labels: Dictionary to rename the legend labels. If `None`, the original labels will be used.
                For categoricals, the keys should be the categories. For numericals, the key should be the column name.
            legend_subtitles: If `True`, subtitles will be added to the legend.
            legend_subtitles_names: Dictionary to rename the legend subtitles. If `None`, the original labels will be used.
                The keys should be the column names.
            show: If `True`, the plot will be shown. If `False`, plotting handels are returned.
            ax: If `None`, a new figure and axes will be created. If an axes object is provided, the plot will be added to it.
            fontsize: Fontsize for the text in the plot.
            subplots_kwargs: Additional keyword arguments for the subplots.
            legend_kwargs: Additional keyword arguments for the legend.

        Examples:
                >>> import ehrdata as ed
                >>> import ehrapy as ep
                >>> edata = ed.dt.diabetes_130_fairlearn(columns_obs_only=["gender", "race", "num_procedures"])
                >>> cohort_tracker = ep.tl.CohortTracker(edata, categorical=["gender", "race"])
                >>> cohort_tracker(edata, "Initial Cohort")
                >>> edata = edata[:1000]
                >>> cohort_tracker(edata, "Filtered Cohort")
                >>> cohort_tracker.plot_cohort_barplot(
                ...     subfigure_title=True,
                ...     color_palette="tab20",
                ...     yticks_labels={
                ...         "race": "Race [%]",
                ...         "gender": "Gender [%]",
                ...         "num_procedures": "#Procedures [mean (stdev)]",
                ...     },
                ...     legend_labels={
                ...         "Unknown/Invalid": "Unknown",
                ...         "num_procedures": "#Procedures",
                ...     },
                ...     legend_kwargs={"bbox_to_anchor": (1, 1.4)},
                ... )

            .. image:: /_static/docstring_previews/cohort_tracking.png
        """
        legend_labels = {} if legend_labels is None else legend_labels
        self._check_legend_labels(legend_labels)

        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs

        legend_subtitles_names = {} if legend_subtitles_names is None else legend_subtitles_names
        self._check_legend_subtitle_names(legend_subtitles_names)

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
                single_ax.set_title(self._tracked_text[idx], size=fontsize)

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
                                size=fontsize,
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
                        size=fontsize,
                    )
                    if idx == 0:
                        name = legend_labels[col] if col in legend_labels.keys() else col

                        legend_handles.append([Patch(color=stacked_bar_color, label=name)])

            single_ax.set_yticks(range(len(self.columns)))
            names = [
                yticks_labels[col] if yticks_labels is not None and col in yticks_labels.keys() else col
                for col in self.columns
            ]
            single_ax.set_yticklabels(names, fontsize=fontsize)

        # These list of lists is needed to reverse the order of the legend labels,
        # making the plot much more readable
        legend_handles.reverse()

        tot_legend_kwargs = {"loc": "best", "bbox_to_anchor": (1, 1), "fontsize": fontsize}
        if legend_kwargs is not None:
            tot_legend_kwargs.update(legend_kwargs)

        def create_legend_with_subtitles(patches_list, subtitles_list, tot_legend_kwargs):
            """Create a legend with subtitles."""
            size = {"size": tot_legend_kwargs["fontsize"]}
            subtitle_font = FontProperties(weight="bold", **size)
            handles = []
            labels = []

            # there can be empty lists which distort the logic of matching patches to subtitles
            patches_list = [patch for patch in patches_list if patch]

            for patches, subtitle in zip(patches_list, subtitles_list, strict=False):
                handles.append(Line2D([], [], linestyle="none", marker="", alpha=0))  # Placeholder for title
                labels.append(subtitle)

                for patch in patches:
                    handles.append(patch)
                    labels.append(patch.get_label())

                # empty space after block
                handles.append(Line2D([], [], linestyle="none", marker="", alpha=0))
                labels.append("")

            legend = axes[0].legend(handles, labels, **tot_legend_kwargs)

            for text in legend.get_texts():
                if text.get_text() in subtitles_list:
                    text.set_font_properties(subtitle_font)

        if legend_subtitles:
            subtitles = [
                legend_subtitles_names[col] if col in legend_subtitles_names.keys() else col
                for col in self.columns[::-1]
            ]
            create_legend_with_subtitles(
                legend_handles,
                subtitles,
                tot_legend_kwargs,
            )
        else:
            legend_handles = [item for sublist in legend_handles for item in sublist]
            plt.legend(handles=legend_handles, **tot_legend_kwargs)

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
        *,
        title: str | None = None,
        arrow_size: float = 0.7,
        show: bool = True,
        ax: Axes | None = None,
        bbox_kwargs: dict | None = None,
        arrowprops_kwargs: dict | None = None,
    ) -> None | list[Axes] | tuple[Figure, list[Axes]]:
        """Flowchart over the tracked steps.

        Create a simple flowchart of data preparation steps tracked with `CohortTracker`.

        Args:
            title: Title of the flow chart.
            arrow_size: The size of the arrows in the plot.
            show: If `True`, the plot will be displayed. If `False`, plotting handels are returned.
            ax: If `None`, a new figure and axes will be created. If an axes object is provided, the plot will be added to it.
            bbox_kwargs: Additional keyword arguments for the node boxes.
            arrowprops_kwargs: Additional keyword arguments for the arrows.

        Examples:
                >>> import ehrdata as ed
                >>> import ehrapy as ep
                >>> edata = ed.dt.diabetes_130_fairlearn(columns_obs_only=["gender", "race"])
                >>> cohort_tracker = ep.tl.CohortTracker(edata)
                >>> cohort_tracker(edata, label="Initial Cohort")
                >>> edata = edata[:1000]
                >>> cohort_tracker(edata, label="Reduced Cohort", operations_done="filtered to first 1000 entries")
                >>> edata = edata[:500]
                >>> cohort_tracker(
                ...     edata,
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
        for _, (y, label) in enumerate(zip(y_positions, node_labels, strict=False)):
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
