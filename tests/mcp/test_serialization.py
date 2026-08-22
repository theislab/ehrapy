"""Tests for 3-tier serialization, column profile ranking, and payload guards (T2, T6, T10)."""

import ehrdata.dt as dt
import numpy as np
import pandas as pd
import pytest

from ehrapy.mcp.serialization import (
    _rank_columns,
    _sample_rows,
    serialize_result,
)


def test_tier1_pass_through_small_dataframe() -> None:
    df = pd.DataFrame({"patient_id": ["p1", "p2", "p3"], "age": [25, 40, 65], "outcome": [0, 1, 0]})
    meta, md = serialize_result(df)
    assert meta["tier"] == 1
    assert "| patient_id | age | outcome |" in md
    assert "| p1 | 25 | 0 |" in md


def test_tier2_column_profile_large_dataframe() -> None:
    # 500 rows, 15 columns
    rng = np.random.default_rng(42)
    data = {f"feat_{i}": rng.standard_normal(500) for i in range(15)}
    df = pd.DataFrame(data)
    meta, md = serialize_result(df)
    assert meta["tier"] == 2
    assert meta["type"] == "dataframe_profile"
    assert "| column | dtype | non-null % | unique | summary |" in md


def test_tier2_column_ranking_weights() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "all_same": [1.0] * 100,
            "high_missing": [np.nan] * 50 + [1.0] * 50,
            "target_col": rng.standard_normal(100),
        }
    )
    relevant = {"target_col"}
    ranked = _rank_columns(df, relevant)
    ranked_cols = [col for col, _ in ranked]
    # target_col (relevance weight 2.0) should rank highest
    assert ranked_cols[0] == "target_col"


def test_tier3_sample_rows_detailed_format() -> None:
    df = pd.DataFrame({"patient_id": [f"p{i}" for i in range(100)], "value": range(100)})
    meta, md = serialize_result(df, response_format="detailed")
    assert meta["tier"] == 3
    assert "| sample | patient_id | value |" in md
    assert "head" in md
    assert "tail" in md


def test_sample_rows_duplicate_indices() -> None:
    """Ensure _sample_rows handles non-unique index labels positionally without corruption (issue #16)."""
    # 100 rows with all index labels set to 0 and a categorical stratification column
    df = pd.DataFrame(
        {"feat": range(100), "category": ["A", "B"] * 50},
        index=[0] * 100,
    )
    sampled = _sample_rows(df, {"category"}, row_budget=20)
    assert len(sampled) == 20
    assert list(sampled.columns) == ["sample", "feat", "category"]
    assert list(sampled["sample"].iloc[:5]) == ["head"] * 5
    assert list(sampled["sample"].iloc[-5:]) == ["tail"] * 5
    assert list(sampled["sample"].iloc[5:15]) == ["sample"] * 10
    assert not sampled.isna().any().any()


def test_empty_dataframe_guard() -> None:
    edata = dt.mimic_2()
    empty_df = pd.DataFrame()
    meta, md = serialize_result(empty_df, function="obs_df", edata=edata)
    assert meta["status"] == "ok_empty"
    assert "Specify columns using `params={'keys': [...]}`" in meta["agent_action"]


def test_whole_table_to_pandas_guard() -> None:
    df = pd.DataFrame({"a": range(10), "b": range(10)})
    meta, md = serialize_result(df, function="to_pandas")
    assert meta["tier"] == 2
    assert "Full table not returned — use `export_edata`" in md


def test_hard_payload_cap_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EHRAPY_MCP_MAX_RESULT_CHARS", "1000")
    large_df = pd.DataFrame({f"col_{i}": [f"val_{j}_{i}" for j in range(100)] for i in range(30)})
    meta, md = serialize_result(large_df)
    assert len(md) <= 1200
    assert "Output truncated" in md
