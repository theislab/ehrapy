import copy
from typing import Any, Union

import graphviz
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scanpy import AnnData
from tableone import TableOne


def _check_columns_exist(df, columns):
    if not all(col in df.columns for col in columns):
        missing_columns = [col for col in columns if col not in df.columns]
        raise ValueError(f"Columns {missing_columns} not found in dataframe.")


def get_column_structure(df, columns):
    columns = df.columns if columns is None else columns
    column_structure = {}

    for column in columns:
        if isinstance(df[column], pd.CategoricalDtype):
            column_structure[column] = {category: [] for category in df[column].cat.categories}
        elif pd.api.types.is_numeric_dtype(df[column]):
            column_structure[column] = []
        else:
            # Coerce to categorical
            df[column] = df[column].astype("category")
            column_structure[column] = {category: [] for category in df[column].cat.categories}
    return column_structure


def log_from_tableone():
    pass  # roughly get_column_dicts I think


class PopulationLogger:
    def __init__(self, adata: AnnData, columns: list = None, *args: Any):
        """
        TODO: write docsring
        """
        if columns is not None:
            _check_columns_exist(adata.obs, columns)

        self.columns = columns if columns is not None else adata.obs.columns

        self.log = get_column_structure(adata.obs, columns)

        self._logged_steps: int = 0

        self._logged_text: list = []

        self._logged_operations: list = []

        self._log_backup = copy.deepcopy(self.log)

        self.columns = args

    def __call__(
        self, adata: AnnData, label: str = None, operations_done: str = None, *args: Any, **tableone_kwargs: Any
    ) -> Any:
        _check_columns_exist(adata, self.columns)

        # log a small text with each logging step, for the flowchart
        log_text = label if label is not None else f"Step {self.logged_steps}"
        log_text += "\n (n=" + str(adata.n_obs) + ")"
        self._logged_text.append(log_text)

        # log a small text with the operations done
        self._logged_operations.append(operations_done)

        self._logged_steps += 1

        t1 = TableOne(adata.obs, **tableone_kwargs)
        # log new stuff
        self._get_column_dicts(t1)

    def _get_column_dicts(self, table_one):
        for key, value in self.log.items():
            if isinstance(value, dict):
                self._get_cat_dicts(table_one, key)
            else:
                # self.log[key] = self.get_num_dicts(table_one, key)
                pass

    def _get_cat_dicts(self, table_one, col):
        for cat in self.log[col].keys():
            pct = float(table_one.cat_table["Overall"].loc[(col, cat)].split("(")[1].split(")")[0])
            self.log[col][cat].append(pct)

    def _get_num_dicts(self, table_one, col):
        return 0  # TODO

    def reset(self):
        self.log = self._log_backup
        self._logged_steps = 0
        self._logged_text = []

    @property
    def logged_steps(self):
        return self._logged_steps

    def plot_population_change(self, save: str = None, return_plot: bool = False):
        """
        Plot the population change over the logged steps.
        TODO: write docstring
        """
        # Plotting
        fig, axes = plt.subplots(self.logged_steps, 1, figsize=(7, 7))

        legend_labels = []

        # if only one step is logged, axes object is not iterable
        if self.logged_steps == 1:
            axes = [axes]

        # each logged step is a subplot
        for idx, ax in enumerate(axes):
            # TODO: continue here
            for pos, (_cols, data) in enumerate(self.log.items()):
                data = pd.DataFrame(data).loc[idx]

                cumwidth = 0

                # Adjust the hue shift based on the category position such that the colors are more distinguishable
                hue_shift = (pos + 1) / len(data)
                colors = sns.color_palette("husl", len(data))
                adjusted_colors = [((color[0] + hue_shift) % 1, color[1], color[2]) for color in colors]

                for i, value in enumerate(data):
                    # value = value[1].values[0]  # Take the value based on idx
                    ax.barh(pos, value, left=cumwidth, color=adjusted_colors[i], height=0.8)

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

        # makes the frames invisible
        # for ax in axes:
        #     ax.axis('off')

        # Add legend for the first subplot
        plt.legend(legend_labels, loc="best", bbox_to_anchor=(-0.5, 1))

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

    def plot_flowchart(self, save: str = None, return_plot: bool = False):
        """
        Plot the flowchart of the logged steps.
        """

        # Create Digraph object
        dot = graphviz.Digraph()

        # Define nodes (edgy nodes)
        for i, text in enumerate(self._logged_text):
            dot.node(name=str(i), label=text, style="filled", shape="box")

        for i, op in enumerate(self._logged_operations[1:]):
            dot.edge(str(i), str(i + 1), label=op, labeldistance="10.5")

        # Render the graph
        dot.render("flow_diagram_edgy_nodes", format="png", cleanup=True)

        return dot
