from __future__ import annotations

import pandas as pd
import pytest
from ehrdata import EHRData

import ehrapy as ep


def test_stratified_table_one_returns_none_inplace(edata_mini):
    out = ep.tl.stratified_table_one(
        edata_mini,
        groupby="station",
        columns=["glucose", "weight", "disease"],
        nonnormal=["glucose"],
    )
    assert out is None
    assert "stratified_table_one" in edata_mini.uns
    assert isinstance(edata_mini.uns["stratified_table_one"]["table"], pd.DataFrame)


def test_stratified_table_one_stores_results(edata_mini):
    ep.tl.stratified_table_one(
        edata_mini,
        groupby="station",
        columns=["glucose", "weight", "disease"],
        nonnormal=["glucose"],
    )
    res = edata_mini.uns["stratified_table_one"]
    assert res["groupby"] == "station"
    assert set(res["groups"]) == {"ICU", "MICU"}
    assert set(res["categorical"]) == {"disease"}
    assert set(res["pvalues"].keys()) == {"glucose", "weight", "disease"}
    for v in res["pvalues"].values():
        assert v == "<0.001" or len(v.split(".")[-1]) == 3
    assert sum(res["group_counts"].values()) == edata_mini.n_obs


def test_stratified_table_one_custom_key(edata_mini):
    ep.tl.stratified_table_one(
        edata_mini,
        groupby="station",
        columns=["glucose", "disease"],
        key_added="my_table",
    )
    assert "my_table" in edata_mini.uns
    assert "stratified_table_one" not in edata_mini.uns


def test_stratified_table_one_copy(edata_mini):
    out = ep.tl.stratified_table_one(
        edata_mini,
        groupby="station",
        columns=["glucose", "disease"],
        copy=True,
    )
    assert isinstance(out, EHRData)
    assert "stratified_table_one" in out.uns
    assert "stratified_table_one" not in edata_mini.uns


def test_stratified_table_one_invalid_groupby(edata_mini):
    with pytest.raises(ValueError, match="not found in edata.obs"):
        ep.tl.stratified_table_one(edata_mini, groupby="does_not_exist")


def test_stratified_table_one_single_group(edata_mini):
    edata_mini.obs["constant"] = "x"
    with pytest.raises(ValueError, match="fewer than 2 unique"):
        ep.tl.stratified_table_one(edata_mini, groupby="constant", columns=["glucose"])


def test_stratified_table_one_invalid_columns(edata_mini):
    with pytest.raises(ValueError, match="not found in edata.obs"):
        ep.tl.stratified_table_one(edata_mini, groupby="station", columns=["glucose", "no_such_column"])


def test_stratified_table_one_requires_ehrdata():
    with pytest.raises(ValueError, match="must be an EHRData"):
        ep.tl.stratified_table_one(
            pd.DataFrame({"a": [1, 2], "g": ["x", "y"]}),  # type: ignore[arg-type]
            groupby="g",
        )


def test_stratified_table_one_three_groups(edata_mini):
    """Three-group comparison should use ANOVA / Kruskal-Wallis / chi-square."""
    ep.tl.stratified_table_one(
        edata_mini,
        groupby="disease",
        columns=["glucose", "weight", "station"],
        nonnormal=["glucose"],
    )
    res = edata_mini.uns["stratified_table_one"]
    assert set(res["groups"]) == {"A", "B", "C"}
    assert set(res["pvalues"].keys()) == {"glucose", "weight", "station"}
