import copy
from typing import Any, Union

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scanpy import AnnData
from tableone import TableOne


def _check_columns_exist(df, columns):
    if not all(col in df.columns for col in columns):
        missing_columns = [col for col in columns if col not in df.columns]
        raise ValueError(f"Columns {missing_columns} not found in dataframe.")


# from tableone: https://github.com/tompollard/tableone/blob/bfd6fbaa4ed3e9f59e1a75191c6296a2a80ccc64/tableone/tableone.py#L555
def _detect_categorical_columns(data) -> list:
    # assume all non-numerical and date columns are categorical
    numeric_cols = set(data._get_numeric_data().columns.values)
    date_cols = set(data.select_dtypes(include=[np.datetime64]).columns)
    likely_cat = set(data.columns) - numeric_cols
    # mypy absolutely looses it if likely_cat is overwritten to be a list
    likely_cat_no_dates = list(likely_cat - date_cols)

    # check proportion of unique values if numerical
    for var in data._get_numeric_data().columns:
        likely_flag = 1.0 * data[var].nunique() / data[var].count() < 0.005
        if likely_flag:
            likely_cat_no_dates.append(var)
    return likely_cat_no_dates


class CohortTracker:
    def __init__(self, adata: AnnData, columns: list = None, categorical: list = None, *args: Any):
        """Track cohort changes over multiple filtering or processing steps.

        This class offers functionality to track and plot cohort changes over multiple filtering or processing steps,
        enabling the user to monitor the impact of each step on the cohort.

        Tightly interacting with the `tableone` package [1].

        categorical : list, optional
        List of columns that contain categorical variables.

        References
        ----------
        [1] Tom Pollard, Alistair E.W. Johnson, Jesse D. Raffa, Roger G. Mark; tableone: An open source Python package for producing summary statistics for research papers, Journal of the American Medical Informatics Association, Volume 24, Issue 2, 1 March 2017, Pages 267–271, https://doi.org/10.1093/jamia/ocw117

        """
        if columns is not None:
            _check_columns_exist(adata.obs, columns)
            _check_columns_exist(adata.obs, categorical)

        self._tracked_steps: int = 0
        self._tracked_text: list = []
        self._tracked_operations: list = []

        self.columns = columns if columns is not None else adata.obs.columns

        # if categorical columns specified, use them
        # else, follow tableone's logic
        self.categorical = categorical if categorical is not None else _detect_categorical_columns(adata.obs[columns])
        self.track = self._get_column_structure(adata.obs)

        self._track_backup = copy.deepcopy(self.track)

    def __call__(
        self, adata: AnnData, label: str = None, operations_done: str = None, *args: Any, **tableone_kwargs: Any
    ) -> Any:
        _check_columns_exist(adata.obs, self.columns)

        # track a small text with each tracking step, for the flowchart
        track_text = label if label is not None else f"Cohort {self.tracked_steps}"
        track_text += "\n (n=" + str(adata.n_obs) + ")"
        self._tracked_text.append(track_text)

        # track a small text with the operations done
        self._tracked_operations.append(operations_done)

        self._tracked_steps += 1

        t1 = TableOne(adata.obs, categorical=self.categorical, **tableone_kwargs)
        # track new stuff
        self._get_column_dicts(t1)

    def _get_column_structure(self, df):
        column_structure = {}
        for column in self.columns:
            if column in self.categorical:
                # if e.g. a column containing integers is deemed categorical, coerce it to categorical
                df[column] = df[column].astype("category")
                column_structure[column] = {category: [] for category in df[column].cat.categories}
            else:
                column_structure[column] = []

        return column_structure

    def _get_column_dicts(self, table_one):
        for col, value in self.track.items():
            if isinstance(value, dict):
                self._get_cat_dicts(table_one, col)
            else:
                self._get_num_dicts(table_one, col)

    def _get_cat_dicts(self, table_one, col):
        for cat in self.track[col].keys():
            # if tableone does not have the category of this column anymore, set the percentage to 0
            # for categorized columns (e.g. gender 1.0/0.0), str(cat) helps to avoid considering the category as a float
            if (col, str(cat)) in table_one.cat_table["Overall"].index:
                pct = float(table_one.cat_table["Overall"].loc[(col, str(cat))].split("(")[1].split(")")[0])
            else:
                pct = 0
            self.track[col][cat].append(pct)

    def _get_num_dicts(self, table_one, col):
        summary = table_one.cont_table["Overall"].loc[(col, "")]
        self.track[col].append(summary)

    def reset(self):
        self.track = self._track_backup
        self._tracked_steps = 0
        self._tracked_text = []

    @property
    def tracked_steps(self):
        return self._tracked_steps

    def plot_cohort_change(
        self,
        set_axis_labels=True,
        subfigure_title: bool = False,
        sns_color_palette: str = "husl",
        save: str = None,
        return_plot: bool = False,
        subplots_kwargs: dict = None,
        legend_kwargs: dict = None,
    ):
        """Plot the cohort change over the tracked steps.

        Create stacked bar plots to monitor cohort changes over the steps tracked with `CohortTracker`.

        Args:
            set_axis_labels: If `True`, the y-axis labels will be set to the column names.
            subfigure_title: If `True`, each subplot will have a title with the `label` provided during tracking.
            sns_color_palette: The color palette to use for the plot. Default is "husl".
            save: If a string is provided, the plot will be saved to the path specified.
            return_plot: If `True`, the plot will be returned as a tuple of (fig, ax).
            subplot_kwargs: Additional keyword arguments for the subplots.
            legend_kwargs: Additional keyword arguments for the legend.

        Returns:
            If `return_plot` a :class:`~matplotlib.figure.Figure` and a :class:`~matplotlib.axes.Axes` or a list of it.

        Example:
            .. code-block:: python

                import ehrapy as ep

                adata = ep.dt.diabetes_130(columns_obs_only=["gender", "race", "weight", "age"])
                cohort_tracker = ep.tl.CohortTracker(adata)
                cohort_tracker(adata, label="original")
                adata = adata[:1000]
                cohort_tracker(adata, label="filtered cohort", operations_done="filtered to first 1000 entries")
                cohort_tracker.plot_cohort_change()
        Preview:
            .. image:: /_static/docstring_previews/flowchart.png
        """
        # Plotting
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs

        fig, axes = plt.subplots(self.tracked_steps, 1, **subplots_kwargs)

        legend_labels = []

        # if only one step is tracked, axes object is not iterable
        if self.tracked_steps == 1:
            axes = [axes]

        # each tracked step is a subplot
        for idx, ax in enumerate(axes):
            if subfigure_title:
                ax.set_title(self._tracked_text[idx])

            # iterate over the tracked columns in the dataframe
            for pos, (_cols, data) in enumerate(self.track.items()):
                data = pd.DataFrame(data).loc[idx]

                cumwidth = 0

                # Adjust the hue shift based on the category position such that the colors are more distinguishable
                hue_shift = (pos + 1) / len(data)
                colors = sns.color_palette(sns_color_palette, len(data))
                adjusted_colors = [((color[0] + hue_shift) % 1, color[1], color[2]) for color in colors]

                # for categoricals, plot multiple bars
                if _cols in self.categorical:
                    for i, value in enumerate(data):
                        ax.barh(pos, value, left=cumwidth, color=adjusted_colors[i], height=0.7)

                        if value > 5:
                            # Add proportion numbers to the bars
                            width = value
                            ax.text(
                                cumwidth + width / 2,
                                pos,
                                f"{value:.1f}",
                                ha="center",
                                va="center",
                                color="white",
                                fontweight="bold",
                            )

                        ax.set_yticks([])
                        ax.set_xticks([])
                        cumwidth += value
                        legend_labels.append(data.index[i])

                # for numericals, plot a single bar
                else:
                    ax.barh(pos, 100, left=cumwidth, color=adjusted_colors[0], height=0.8)
                    ax.text(
                        100 / 2,
                        pos,
                        data[0],
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                    )
                    legend_labels.append(_cols)

            # Set y-axis labels
            if set_axis_labels:
                ax.set_yticks(
                    range(len(self.track.keys()))
                )  # Set ticks at positions corresponding to the number of columns
                ax.set_yticklabels(self.track.keys())  # Set y-axis labels to the column names

            # makes the frames invisible
            # for ax in axes:
            #     ax.axis('off')

        # Add legend
        tot_legend_kwargs = {"loc": "best", "bbox_to_anchor": (1, 1)}
        tot_legend_kwargs.update(legend_kwargs)

        plt.legend(legend_labels, **tot_legend_kwargs)

        if save is not None:
            if not isinstance(save, str):
                raise ValueError("'save' must be a string.")
            plt.savefig(
                save,
            )

        if return_plot:
            return fig, axes

        else:
            plt.tight_layout()
            plt.show()

    def plot_flowchart(self, save: str = None, return_plot: bool = True):
        """Flowchart over the tracked steps.

        Create a simple flowchart of data preparation steps tracked with `CohortTracker`.

        Args:
            save: If a string is provided, the plot will be saved to the path specified.
            return_plot: If `True`, the plot will be returned as a :class:`~graphviz.Digraph`.

        Returns:
            If `return_plot` a :class:`~graphviz.Digraph`.

        Example:
            .. code-block:: python

                import ehrapy as ep

                adata = ep.dt.diabetes_130(columns_obs_only=["gender", "race", "weight", "age"])
                cohort_tracker = ep.tl.CohortTracker(adata)
                cohort_tracker(adata, label="original")
                adata = adata[:1000]
                cohort_tracker(adata, label="filtered cohort", operations_done="filtered to first 1000 entries")
                cohort_tracker.plot_flowchart()
        Preview:
            .. image:: /_static/docstring_previews/flowchart.png

        """

        # Create Digraph object
        dot = graphviz.Digraph()

        # Define nodes (edgy nodes)
        for i, text in enumerate(self._tracked_text):
            dot.node(name=str(i), label=text, style="filled", shape="box")

        for i, op in enumerate(self._tracked_operations[1:]):
            dot.edge(str(i), str(i + 1), label=op, labeldistance="2.5")

        # Render the graph
        if save is not None:
            if not isinstance(save, str):
                raise ValueError("'save' must be a string.")
            dot.render(save, format="png", cleanup=True)

        # Think that to be shown, the plot can a) be rendered (as above) or be "printed" by the notebook
        if return_plot:
            return dot
