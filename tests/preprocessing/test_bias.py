import ehrdata as ed
import numpy as np
import pandas as pd
import pytest
from ehrdata.core.constants import CATEGORICAL_TAG, DEFAULT_TEM_LAYER_NAME, FEATURE_TYPE_KEY

import ehrapy as ep


def test_detect_bias_all_sensitive_features(edata_small_bias):
    results = ep.pp.detect_bias(
        edata_small_bias, "all", run_feature_importances=True, corr_method="spearman", feature_importance_threshold=0.4
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


def test_explicit_impute_3D_edata(edata_blob_small):
    ep.pp.detect_bias(edata_blob_small, sensitive_features=["feature_1"], layer="layer_2")
    with pytest.raises(ValueError, match=r"only supports 2D data"):
        ep.pp.detect_bias(edata_blob_small, sensitive_features=["feature_1"], layer=DEFAULT_TEM_LAYER_NAME)


def test_detect_bias_specified_sensitive_features(edata_small_bias):
    results, result_adata = ep.pp.detect_bias(
        edata_small_bias,
        ["contin1", "cat1"],
        run_feature_importances=True,
        corr_method="spearman",
        feature_importance_threshold=0.5,
        prediction_confidence_threshold=0.4,
        copy=True,
    )

    assert "smd" not in edata_small_bias.uns.keys()
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
    edata = ed.io.from_pandas(
        pd.DataFrame({"Unencoded": ["A", "B", "C", "D", "E", "F"], "Encoded": [1, 2, 3, 4, 5, 6]})
    )
    edata.var[FEATURE_TYPE_KEY] = [CATEGORICAL_TAG] * 2

    with pytest.raises(ValueError):
        ep.pp.detect_bias(edata, "all")
