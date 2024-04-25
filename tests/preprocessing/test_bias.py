import numpy as np
import pandas as pd
import pytest

import ehrapy as ep
from ehrapy.anndata._constants import CATEGORICAL_TAG, CONTINUOUS_TAG, FEATURE_TYPE_KEY


@pytest.fixture
def adata():
    corr = np.random.randint(0, 100, 100)
    df = pd.DataFrame(
        {
            "corr1": corr,
            "corr2": corr * 2,
            "corr3": corr * -1,
            "contin1": np.random.randint(0, 20, 50).tolist() + np.random.randint(20, 40, 50).tolist(),
            "cat1": [0] * 50 + [1] * 50,
            "cat2": [10] * 10 + [11] * 40 + [10] * 30 + [11] * 20,
        }
    )
    adata = ep.ad.df_to_anndata(df)
    adata.var[FEATURE_TYPE_KEY] = [CONTINUOUS_TAG] * 4 + [CATEGORICAL_TAG] * 2
    return adata


def test_detect_bias_all_sens_features(adata):
    results = ep.pp.detect_bias(
        adata, "all", run_feature_importances=True, corr_method="spearman", feature_importance_threshold=0.4
    )

    assert "feature_correlations" in results.keys()
    df = results["feature_correlations"]
    assert len(df) == 4
    assert df[(df["Feature 1"] == "corr1") & (df["Feature 2"] == "corr2")]["Spearman CC"].values[0] == 1
    assert df[(df["Feature 1"] == "corr1") & (df["Feature 2"] == "corr3")]["Spearman CC"].values[0] == -1
    assert df[(df["Feature 1"] == "corr2") & (df["Feature 2"] == "corr3")]["Spearman CC"].values[0] == -1
    assert df[(df["Feature 1"] == "contin1") & (df["Feature 2"] == "cat1")]["Spearman CC"].values[0] > 0.5

    assert "standardized_mean_differences" in results.keys()
    df = results["standardized_mean_differences"]
    print(df)
    # TODO

    assert "categorical_value_counts" in results.keys()
    df = results["categorical_value_counts"]
    assert len(df) == 4
    assert df[(df["Sensitive Feature"] == "cat1") & (df["Sensitive Group"] == 0)]["Group 1 Percentage"].values[0] == 0.2
    assert df[(df["Sensitive Feature"] == "cat1") & (df["Sensitive Group"] == 0)]["Group 2 Percentage"].values[0] == 0.8

    assert "feature_importances" in results.keys()
    df = results["feature_importances"]
    assert len(df) == 7  # 6 for the pairwise correlating features and one for contin1, which predicts cat1


def test_detect_bias_specific_sens_features(adata):
    results = ep.pp.detect_bias(
        adata,
        ["contin1", "cat1"],
        run_feature_importances=True,
        corr_method="spearman",
        feature_importance_threshold=0.4,
    )

    assert "feature_correlations" in results.keys()
    results["feature_correlations"]
    # TODO: Add actual tests
