"""Tests for unknown argument middleware, steering suggestions, prompts, and provenance (T5, T17, T18, T19)."""

from __future__ import annotations

import asyncio
import json

from fastmcp import Client

from ehrapy.mcp.edata_store import load_edata
from ehrapy.mcp.server import mcp


def _run(coro):
    return asyncio.run(coro)


def test_argument_validation_middleware_rejects_unknown_args() -> None:
    """A tool without a `params` dict rejects unknown arguments loudly (T5, origin #3)."""

    async def _test():
        async with Client(mcp) as client:
            res = await client.call_tool(
                "get_edata_snapshot",
                {"unknown_bogus_arg": 123},
            )
            assert res.structured_content["status"] == "error"
            assert res.structured_content["error_code"] == "UNKNOWN_ARGUMENT"
            assert "unknown_bogus_arg" in res.structured_content["details"]["unknown_arguments"]

    _run(_test())


def test_argument_validation_middleware_folds_kwargs_into_params() -> None:
    """A tool with a `params` dict folds stray top-level kwargs into it (origin #3)."""

    async def _test():
        async with Client(mcp) as client:
            await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
            res = await client.call_tool(
                "run_get",
                {"function": "obs_df", "keys": ["age"]},
            )
            struct = res.structured_content
            assert struct["status"] == "ok", struct
            assert struct["folded_arguments"] == ["keys"]

    _run(_test())


def test_explicit_params_win_over_folded_kwarg() -> None:
    """An explicit params entry is not clobbered by a folded top-level kwarg of the same name."""

    async def _test():
        async with Client(mcp) as client:
            await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
            res = await client.call_tool(
                "run_get",
                {"function": "obs_df", "params": {"keys": ["age"]}, "keys": ["bmi"]},
            )
            assert res.structured_content["status"] == "ok"
            # params={'keys': ['age']} wins, so the rendered table is the age column.
            text = " ".join(getattr(c, "text", "") for c in res.content)
            assert "age" in text

    _run(_test())


def test_argument_validation_middleware_strips_wait_for_previous() -> None:
    """Test that middleware strips wait_for_previous without failing (T5)."""

    async def _test():
        async with Client(mcp) as client:
            res = await client.call_tool(
                "get_runtime_context",
                {"wait_for_previous": True},
            )
            assert res.structured_content["status"] == "ok"

    _run(_test())


def test_steering_suggested_next() -> None:
    """Test that steering suggestions are included in structured_content and content (T17)."""

    async def _test():
        async with Client(mcp) as client:
            demo_res = await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
            assert "suggested_next" in demo_res.structured_content
            suggestions = demo_res.structured_content["suggested_next"]
            assert any("qc_metrics" in s for s in suggestions)
            assert "Suggested next:" in demo_res.content[0].text

    _run(_test())


def test_mcp_prompts() -> None:
    """Test that MCP prompts are registered and return instructions (T18)."""

    async def _test():
        async with Client(mcp) as client:
            prompts = await client.list_prompts()
            prompt_names = [p.name for p in prompts]
            assert "ehrapy-explore" in prompt_names
            assert "ehrapy-clustering" in prompt_names
            assert "ehrapy-survival" in prompt_names

            res = await client.get_prompt("ehrapy-survival", {"dataset": "mimic_2"})
            assert "kaplan_meier" in res.messages[0].content.text

    _run(_test())


def test_provenance_op_log() -> None:
    """Test that mutating operations append to edata.uns['ehrapy_mcp_ops'] (T19)."""

    async def _test():
        async with Client(mcp) as client:
            demo_res = await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
            edata_id = demo_res.structured_content["edata_id"]

            await client.call_tool(
                "run_preprocessing",
                {"function": "qc_metrics", "edata_id": edata_id, "params": {"qc_vars": []}},
            )

            # Load dataset from store
            edata = load_edata(edata_id)
            assert "ehrapy_mcp_ops" in edata.uns
            ops = [json.loads(op) if isinstance(op, str) else op for op in edata.uns["ehrapy_mcp_ops"]]
            assert len(ops) >= 1
            last_op = ops[-1]
            assert last_op["namespace"] == "preprocessing"
            assert last_op["function"] == "qc_metrics"

    _run(_test())
