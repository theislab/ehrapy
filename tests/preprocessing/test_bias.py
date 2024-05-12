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


def test_detect_bias_all_sensitive_features(adata):
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
    assert len(df) == 4  # Both groups of cat1, cat2 respectively show a high SMD with contin1
    smd_key = "Standardized Mean Difference"
    assert df[(df["Sensitive Feature"] == "cat1") & (df["Sensitive Group"] == 0)][smd_key].values[0] < 0
    assert df[(df["Sensitive Feature"] == "cat1") & (df["Sensitive Group"] == 1)][smd_key].values[0] > 0
    assert df[(df["Sensitive Feature"] == "cat2") & (df["Sensitive Group"] == 10)][smd_key].values[0] > 0
    assert df[(df["Sensitive Feature"] == "cat2") & (df["Sensitive Group"] == 11)][smd_key].values[0] < 0

    assert "categorical_value_counts" in results.keys()
    df = results["categorical_value_counts"]
    assert len(df) == 2
    assert df[(df["Sensitive Feature"] == "cat1") & (df["Sensitive Group"] == 0)]["Group 1 Percentage"].values[0] == 0.2
    assert df[(df["Sensitive Feature"] == "cat1") & (df["Sensitive Group"] == 0)]["Group 2 Percentage"].values[0] == 0.8
    assert (
        df[(df["Sensitive Feature"] == "cat2") & (df["Sensitive Group"] == 10)]["Group 1 Percentage"].values[0] == 0.25
    )
    assert (
        df[(df["Sensitive Feature"] == "cat2") & (df["Sensitive Group"] == 10)]["Group 2 Percentage"].values[0] == 0.75
    )

    assert "feature_importances" in results.keys()
    df = results["feature_importances"]
    assert len(df) >= 7  # 6 for the pairwise correlating features and one/two for contin1, which predicts cat1


def test_detect_bias_specified_sensitive_features(adata):
    results, result_adata = ep.pp.detect_bias(
        adata,
        ["contin1", "cat1"],
        run_feature_importances=True,
        corr_method="spearman",
        feature_importance_threshold=0.5,
        prediction_confidence_threshold=0.4,
        copy=True,
    )

    assert "smd" not in adata.uns.keys()
    assert "smd" in result_adata.uns.keys()

    assert "feature_correlations" in results.keys()
    df = results["feature_correlations"]
    assert len(df) == 2  # cat1 & contin1 and contin1 & cat1
    assert np.all(df["Spearman CC"] > 0.5)

    assert "standardized_mean_differences" in results.keys()
    df = results["standardized_mean_differences"]
    assert len(df) == 2  # Both groups of cat1 show a high SMD with contin1
    smd_key = "Standardized Mean Difference"
    assert df[(df["Sensitive Feature"] == "cat1") & (df["Sensitive Group"] == 0)][smd_key].values[0] < -1
    assert df[(df["Sensitive Feature"] == "cat1") & (df["Sensitive Group"] == 1)][smd_key].values[0] > 1

    assert "categorical_value_counts" in results.keys()
    df = results["categorical_value_counts"]
    assert len(df) == 1
    assert df[(df["Sensitive Feature"] == "cat1") & (df["Sensitive Group"] == 0)]["Group 1 Percentage"].values[0] == 0.2
    assert df[(df["Sensitive Feature"] == "cat1") & (df["Sensitive Group"] == 0)]["Group 2 Percentage"].values[0] == 0.8

    assert "feature_importances" in results.keys()
    df = results["feature_importances"]
    assert len(df) == 2  # contin1 predicts cat1 and cat1 predicts contin1


def test_unencoded_data():
    adata = ep.ad.df_to_anndata(
        pd.DataFrame({"Unencoded": ["A", "B", "C", "D", "E", "F"], "Encoded": [1, 2, 3, 4, 5, 6]})
    )
    adata.var[FEATURE_TYPE_KEY] = [CATEGORICAL_TAG] * 2

    with pytest.raises(ValueError):
        ep.pp.detect_bias(adata, "all")
