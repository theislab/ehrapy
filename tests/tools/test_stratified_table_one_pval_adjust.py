from __future__ import annotations

import ehrapy as ep


def test_stratified_table_one_stores_adjusted_pvalues(edata_mini):
    columns = ["glucose", "weight", "disease"]
    ep.tl.stratified_table_one(edata_mini, groupby="station", columns=columns, pval_adjust="bonferroni")

    res = edata_mini.uns["stratified_table_one"]
    table = res["table"]
    shown = {
        column: table.loc[table["variable"].str.startswith(f"{column}, "), "P-Value (adjusted)"].iloc[0]
        for column in columns
    }
    assert res["pvalues"] == shown == {"glucose": "0.072", "weight": "0.604", "disease": "1.000"}
