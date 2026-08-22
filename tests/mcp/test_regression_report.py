"""Test the exact regression sequence from testing report."""

from __future__ import annotations

import asyncio
import time

from fastmcp import Client

from ehrapy.mcp.server import mcp


def test_regression_sequence() -> None:
    async def _test():
        async with Client(mcp) as client:
            # 1. get_runtime_context returns quickly (<2s)
            t0 = time.time()
            ctx_res = await client.call_tool("get_runtime_context", {})
            t_ctx = time.time() - t0
            assert ctx_res.structured_content["status"] == "ok"
            assert t_ctx < 2.0, f"get_runtime_context took too long: {t_ctx}s"

            # 2. load_demo_dataset succeeds and returns valid handle
            demo_res = await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
            assert demo_res.structured_content["status"] == "ok"
            edata_id = demo_res.structured_content["edata_id"]
            assert edata_id is not None

            # 3. list_ehrapy_functions returns in <2s
            t0 = time.time()
            list_res = await client.call_tool("list_ehrapy_functions", {"namespace": "preprocessing"})
            t_list = time.time() - t0
            assert list_res.structured_content["status"] == "ok"
            assert list_res.structured_content["count"] >= 30
            assert t_list < 2.0, f"list_ehrapy_functions took too long: {t_list}s"

    asyncio.run(_test())
