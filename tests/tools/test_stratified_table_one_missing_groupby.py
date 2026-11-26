from __future__ import annotations

import numpy as np

import ehrapy as ep


def test_stratified_table_one_leaves_out_missing_groups(edata_mini):
    columns = ["glucose", "weight", "disease"]
    edata_mini.obs.loc[edata_mini.obs_names[[0, 6]], "station"] = np.nan
    complete = edata_mini[edata_mini.obs["station"].notna()].copy()

    ep.tl.stratified_table_one(edata_mini, groupby="station", columns=columns)
    ep.tl.stratified_table_one(complete, groupby="station", columns=columns)

    res = edata_mini.uns["stratified_table_one"]
    expected = complete.uns["stratified_table_one"]
    assert res["groups"] == expected["groups"] == ["ICU", "MICU"]
    assert res["group_counts"] == expected["group_counts"]
    assert res["pvalues"] == expected["pvalues"]
    assert "nan" not in res["table"].columns
