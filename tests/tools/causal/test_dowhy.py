import warnings

import anndata
import dowhy
import dowhy.datasets
import matplotlib.pyplot as plt
import numpy as np
import pytest

import ehrapy as ep

warnings.filterwarnings("ignore")


class TestCausal:
    def setup_method(self):
        self.seed = 8
        np.random.seed(8)

        linear_data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=5,
            num_instruments=2,
            num_samples=1000,
            treatment_is_binary=True,
        )
        self.linear_data = anndata.AnnData(linear_data["df"].astype(np.float32))
        self.linear_graph = linear_data["gml_graph"]
        self.outcome_name = "y"
        self.treatment_name = "v0"

    def test_dowhy_linear_dataset(self):
        np.random.seed(self.seed)
        estimate, refute_results = ep.tl.causal_inference(
            adata=self.linear_data,
            graph=self.linear_graph,
            treatment=self.treatment_name,
            outcome=self.outcome_name,
            estimation_method="backdoor.linear_regression",
            return_as="estimate+refute",
        )

        assert isinstance(refute_results, dict)
        assert len(refute_results) == 4
        assert isinstance(estimate, dowhy.causal_estimator.CausalEstimate)
        assert np.round(refute_results["Refute: Add a random common cause"]["test_significance"], 3) == 10.002
        assert np.round(refute_results["Refute: Use a subset of data"]["test_significance"], 3) == 10.002

    def test_causal_inference_input_types(self) -> None:
        # Test if function raises TypeError for invalid input types
        with pytest.raises(TypeError):
            ep.tl.causal_inference(
                adata=123,  # type: ignore
                graph=self.linear_graph,
                treatment=self.treatment_name,
                outcome=self.outcome_name,
                estimation_method="backdoor.propensity_score_matching",
            )

        with pytest.raises(TypeError):
            ep.tl.causal_inference(
                adata=self.linear_data,
                graph=123,
                treatment=self.treatment_name,
                outcome=self.outcome_name,
                estimation_method="backdoor.propensity_score_matching",
            )

        with pytest.raises(TypeError):
            ep.tl.causal_inference(
                adata=self.linear_data,
                graph=self.linear_graph,
                treatment=123,  # type: ignore
                outcome=self.outcome_name,
                estimation_method="backdoor.propensity_score_matching",
            )

        with pytest.raises(TypeError):
            ep.tl.causal_inference(
                adata=self.linear_data,
                graph=self.linear_graph,
                treatment=self.treatment_name,
                outcome=123,  # type: ignore
                estimation_method="backdoor.propensity_score_matching",
            )

        with pytest.raises(TypeError):
            ep.tl.causal_inference(
                adata=self.linear_data,
                graph=self.linear_graph,
                treatment=self.treatment_name,
                outcome=self.outcome_name,
                estimation_method=123,  # type: ignore
            )

        with pytest.raises(ValueError):
            ep.tl.causal_inference(
                adata=self.linear_data,
                graph=self.linear_graph,
                treatment=self.treatment_name,
                outcome=self.outcome_name,
                estimation_method="123",  # type: ignore
            )

        with pytest.raises(TypeError):
            ep.tl.causal_inference(
                adata=self.linear_data,
                graph=self.linear_graph,
                treatment=self.treatment_name,
                outcome=self.outcome_name,
                estimation_method="backdoor.propensity_score_matching",
                refute_methods=["placebo_treatment_refuter", "random_common_cause", 123],  # type: ignore
            )

        with pytest.raises(ValueError):
            ep.tl.causal_inference(
                adata=self.linear_data,
                graph=self.linear_graph,
                treatment=self.treatment_name,
                outcome=self.outcome_name,
                estimation_method="backdoor.propensity_score_matching",
                refute_methods=["placebo_treatment_refuter", "random_common_cause", "123"],  # type: ignore
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
        assert "10.002" in str(ax.get_title())
