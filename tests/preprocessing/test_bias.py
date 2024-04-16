import numpy as np
import pandas as pd
import pytest

import ehrapy as ep


@pytest.fixture
def adata():
    corr = np.random.randint(0, 100, 100)
    df = pd.DataFrame(
        {
            "corr1": corr,
            "corr2": corr * 2,
            "corr3": corr * -1,
            "continuous1": np.random.randint(0, 20, 50).tolist() + np.random.randint(20, 40, 50).tolist(),
            "cat1": [0] * 50 + [1] * 50,
            "cat2": [10] * 10 + [11] * 40 + [10] * 30 + [11] * 20,
        }
    )
    adata = ep.ad.df_to_anndata(df)
    adata.var["feature_type"] = ["continuous"] * 4 + [
        "categorical"
    ] * 2  # TODO: Adjust to use variable for name as specified in _constants
    return adata


def test_detect_bias_all_sens_features(adata):
    results = ep.pp.detect_bias(adata, "all", run_feature_importances=True)

    assert "feature_correlations" in results.keys()
    feature_corrs = results["feature_correlations"]
    assert len(feature_corrs) == 4
    assert (
        feature_corrs[(feature_corrs["Feature 1"] == "corr1") & (feature_corrs["Feature 2"] == "corr2")][
            "Correlation Coefficient"
        ].values[0]
        == 1
    )
    assert (
        feature_corrs[(feature_corrs["Feature 1"] == "corr1") & (feature_corrs["Feature 2"] == "corr3")][
            "Correlation Coefficient"
        ].values[0]
        == -1
    )
    assert (
        feature_corrs[(feature_corrs["Feature 1"] == "corr2") & (feature_corrs["Feature 2"] == "corr3")][
            "Correlation Coefficient"
        ].values[0]
        == -1
    )
    assert (
        feature_corrs[(feature_corrs["Feature 1"] == "continuous1") & (feature_corrs["Feature 2"] == "cat1")][
            "Correlation Coefficient"
        ].values[0]
        > 0.5
    )

    assert "standardized_mean_differences" in results.keys()
    results["standardized_mean_differences"]
    # TODO

    assert "categorical_value_counts" in results.keys()
    cat_value_counts = results["categorical_value_counts"]
    assert len(cat_value_counts) == 4
    assert (
        cat_value_counts[
            (cat_value_counts["Sensitive Feature"] == "cat1") & (cat_value_counts["Sensitive Group"] == 0)
        ]["Group 1 Percentage"].values[0]
        == 0.2
    )
    assert (
        cat_value_counts[
            (cat_value_counts["Sensitive Feature"] == "cat1") & (cat_value_counts["Sensitive Group"] == 0)
        ]["Group 2 Percentage"].values[0]
        == 0.8
    )

    assert "feature_importances" in results.keys()
    feat_importances = results["feature_importances"]
    assert (
        len(feat_importances) == 7
    )  # 6 for the pairwise correlating features and one for continuous1, which predicts cat1
    assert (
        feat_importances[
            (feat_importances["Sensitive Feature"] == "continuous1") & (feat_importances["Predicted Feature"] == "cat1")
        ]["Feature Importance"].values[0]
        == 1
    )


def test_detect_bias_specific_sens_features(adata):
    ep.pp.detect_bias(adata, ["continuous1", "cat1"], run_feature_importances=True)

    # TODO: Add actual tests
