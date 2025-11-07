import re
import warnings

import anndata
import dowhy
import dowhy.datasets
import matplotlib.pyplot as plt
import numpy as np
import pytest
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

import ehrapy as ep

warnings.filterwarnings("ignore")


class TestCausal:
    def setup_method(self):
        linear_data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=5,
            num_instruments=2,
            num_samples=1000,
            treatment_is_binary=True,
        )
        X = linear_data["df"].astype(np.float32)
        R_layer = np.stack([X, X], axis=2)
        self.linear_data = anndata.AnnData(X, layers={DEFAULT_TEM_LAYER_NAME: R_layer})
        self.linear_graph = linear_data["gml_graph"]
        self.outcome_name = "y"
        self.treatment_name = "v0"

    def test_dowhy_linear_dataset(self):
        estimate, refute_results = ep.tl.causal_inference(
            adata=self.linear_data,
            graph=self.linear_graph,
            treatment=self.treatment_name,
            outcome=self.outcome_name,
            estimation_method="backdoor.linear_regression",
            return_as="estimate+refute",
        )

        assert isinstance(refute_results, dict)
        assert len(refute_results) == 6
        assert isinstance(estimate, dowhy.causal_estimator.CausalEstimate)
        assert np.isclose(
            np.round(refute_results["Refute: Add a random common cause"]["test_significance"], 3), 10.002, atol=0.005
        )
        assert np.isclose(
            np.round(refute_results["Refute: Add a random common cause"]["test_significance"], 3), 10.002, atol=0.005
        )

    def test_dowhy_linear_dataset_3D_edata(self):
        self.linear_data.layers["layer_2"] = self.linear_data.X.copy()
        ep.tl.causal_inference(
            edata=self.linear_data,
            graph=self.linear_graph,
            treatment=self.treatment_name,
            outcome=self.outcome_name,
            estimation_method="backdoor.linear_regression",
            layer="layer_2",
        )
        with pytest.raises(ValueError, match=r"only supports 2D data"):
            ep.tl.causal_inference(
                edata=self.linear_data,
                graph=self.linear_graph,
                treatment=self.treatment_name,
                outcome=self.outcome_name,
                estimation_method="backdoor.linear_regression",
                layer=DEFAULT_TEM_LAYER_NAME,
            )

    def test_plot_causal_effect(self):
        estimate = ep.tl.causal_inference(
            adata=self.linear_data,
            graph=self.linear_graph,
            treatment=self.treatment_name,
            outcome=self.outcome_name,
            estimation_method="backdoor.linear_regression",
            return_as="estimate",
            show_graph=False,
            show_refute_plots=False,
        )
        ax = ep.pl.causal_effect(estimate)

        assert isinstance(ax, plt.Axes)
        legend = ax.get_legend()
        assert len(legend.get_texts()) == 2  # Check the number of legend labels
        assert legend.get_texts()[0].get_text() == "Observed data"
        assert legend.get_texts()[1].get_text() == "Causal variation"
        assert re.search(r"(9\.99\d+|10\.0)", str(ax.get_title()))
