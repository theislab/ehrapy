"""CI token-budget gate for MCP tools, schemas, and outputs (T16)."""

from __future__ import annotations

import asyncio
import json

from fastmcp import Client

from ehrapy.mcp.catalog import list_functions, list_namespaces
from ehrapy.mcp.server import mcp


def _run(coro):
    return asyncio.run(coro)


def test_tools_list_payload_budget() -> None:
    """Test that tools/list payload is <= 16KB and <= 4000 tokens."""

    async def _test():
        async with Client(mcp) as client:
            tools = await client.list_tools()
            assert len(tools) == 14

            tools_dicts = [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.inputSchema,
                    "annotations": t.annotations.model_dump() if t.annotations else None,
                }
                for t in tools
            ]
            serialized = json.dumps(tools_dicts)
            byte_size = len(serialized.encode("utf-8"))
            approx_tokens = len(serialized) / 4

            assert byte_size <= 16 * 1024, f"tools/list is {byte_size} bytes (> 16KB)"
            assert approx_tokens <= 4000, f"tools/list is ~{approx_tokens} tokens (> 4000 tokens)"

            # Check individual tool descriptions
            for t in tools:
                desc_bytes = len((t.description or "").encode("utf-8"))
                assert desc_bytes <= 2048, f"Tool {t.name} description exceeds 2KB: {desc_bytes} bytes"
                assert desc_bytes <= 600, f"Tool {t.name} description exceeds 600 bytes: {desc_bytes} bytes"

    _run(_test())


def test_workflow_guide_budget() -> None:
    """Test that get_workflow_guide output is <= 2000 tokens."""

    async def _test():
        async with Client(mcp) as client:
            res = await client.call_tool("get_workflow_guide", {})
            content = res.content[0].text
            approx_tokens = len(content) / 4
            assert approx_tokens <= 2000, f"Workflow guide is ~{approx_tokens} tokens (> 2000 tokens)"

    _run(_test())


def test_function_help_budget_all_namespaces() -> None:
    """Test that get_function_help is <= 2500 chars for representative functions."""

    async def _test():
        async with Client(mcp) as client:
            for ns in list_namespaces():
                funcs = list_functions(ns)
                for fn_name in funcs[:5]:  # test first 5 in each namespace
                    res = await client.call_tool("get_function_help", {"namespace": ns, "function": fn_name})
                    content = res.content[0].text
                    assert len(content) <= 2500, f"Help for {ns}.{fn_name} is {len(content)} chars (> 2500 chars)"

    _run(_test())


def test_qc_metrics_output_budget() -> None:
    """Test that qc_metrics response on mimic_2 is <= 1500 tokens."""

    async def _test():
        async with Client(mcp) as client:
            demo_res = await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
            edata_id = demo_res.structured_content["edata_id"]

            qc_res = await client.call_tool(
                "run_preprocessing",
                {"function": "qc_metrics", "edata_id": edata_id, "params": {"qc_vars": []}},
            )
            content = qc_res.content[0].text
            approx_tokens = len(content) / 4
            assert approx_tokens <= 1500, f"qc_metrics output is ~{approx_tokens} tokens (> 1500 tokens)"

    _run(_test())
