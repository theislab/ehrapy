"""Tests for MCP server robustness, timeouts, demo dataset writable fallbacks, and sandbox error UX."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path  # noqa: TC003
from unittest.mock import patch

import pytest  # noqa: TC002
from fastmcp import Client

import ehrapy as ep
from ehrapy.mcp.dispatch import get_tool_timeout
from ehrapy.mcp.registry import MCPRegistry
from ehrapy.mcp.server import mcp


def _run(coro):
    return asyncio.run(coro)


def test_runtime_context_surfaces_demo_dir_and_writability() -> None:
    """Test that get_runtime_context reports demo_data_dir and cache_writable (Issue 1)."""

    async def _test():
        async with Client(mcp) as client:
            ctx_res = await client.call_tool("get_runtime_context", {})
            assert ctx_res.structured_content["status"] == "ok"
            assert "demo_data_dir" in ctx_res.structured_content
            assert ctx_res.structured_content["cache_writable"] is True
            assert "Cache-Bridge Ingestion" in ctx_res.content[0].text

    _run(_test())


def test_demo_dataset_loads_into_writable_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that load_demo_dataset downloads into configured/writable cache dir (Issue 1)."""
    custom_demo = tmp_path / "demo_cache"
    monkeypatch.setenv("EHRAPY_MCP_DEMO_DATA_DIR", str(custom_demo))

    reg = MCPRegistry()
    assert reg.demo_data_dir() == custom_demo

    async def _test():
        async with Client(mcp) as client:
            demo_res = await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
            assert demo_res.structured_content["status"] == "ok"
            assert demo_res.structured_content["edata_id"] is not None

    _run(_test())


def test_tool_timeout_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that slow tool operations fail fast with structured TIMEOUT error (Issue 2)."""
    monkeypatch.setenv("EHRAPY_MCP_TIMEOUT_SECONDS", "0.3")
    assert get_tool_timeout() == 0.3

    def _mock_slow_qc(*args, **kwargs):
        time.sleep(1.0)

    with patch.object(ep.preprocessing, "qc_metrics", side_effect=_mock_slow_qc):

        async def _test():
            async with Client(mcp) as client:
                demo_res = await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
                edata_id = demo_res.structured_content["edata_id"]

                qc_res = await client.call_tool(
                    "run_preprocessing",
                    {"function": "qc_metrics", "edata_id": edata_id, "params": {"qc_vars": []}},
                )
                assert qc_res.structured_content["status"] == "error"
                assert qc_res.structured_content["error_code"] == "TIMEOUT"
                assert "timed out after 0.3s" in qc_res.structured_content["reason"]

        _run(_test())


def test_sandbox_path_detection_and_cache_bridge_guidance() -> None:
    """Test that sandboxed client paths return HOST_PATH_NOT_VISIBLE with Cache-Bridge guidance (Issue 3)."""

    async def _test():
        async with Client(mcp) as client:
            # 1. Test sandbox prefix (/home/claude)
            res1 = await client.call_tool("ingest_dataset", {"file_path": "/home/claude/cohort.csv"})
            assert res1.structured_content["status"] == "error"
            assert res1.structured_content["error_code"] == "HOST_PATH_NOT_VISIBLE"
            assert "cache_dir" in res1.structured_content["agent_action"]
            assert res1.structured_content["details"]["path_context"]["looks_like_sandbox_path"] is True

            # 2. Test sandbox prefix (/workspace/data.csv)
            res2 = await client.call_tool("ingest_dataset", {"file_path": "/workspace/data.csv"})
            assert res2.structured_content["status"] == "error"
            assert res2.structured_content["error_code"] == "HOST_PATH_NOT_VISIBLE"

            # 3. Test non-sandbox missing file
            res3 = await client.call_tool("ingest_dataset", {"file_path": "/var/data/missing_file_xyz.csv"})
            assert res3.structured_content["status"] == "error"
            assert res3.structured_content["error_code"] == "FILE_NOT_FOUND"
            assert "cache_dir" in res3.structured_content["agent_action"]

    _run(_test())
