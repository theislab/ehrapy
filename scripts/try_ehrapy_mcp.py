"""Quick local smoke test for ehrapy-mcp over stdio."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


async def main() -> None:
    """Exercise the packaged ehrapy-mcp server over stdio: list tools, load, ingest, query."""
    transport = StdioTransport(command=sys.executable, args=["-m", "ehrapy.mcp.server"])
    async with Client(transport=transport) as client:
        tools = await client.list_tools()
        print(f"tools: {len(tools)}")

        guide = (await client.call_tool("get_workflow_guide", {})).content[0].text
        assert "run_preprocessing" in guide
        print("get_workflow_guide: ok")

        demo = (await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})).structured_content
        edata_id = demo["edata_id"]
        print(f"load_demo_dataset: ok ({edata_id})")

        qc = (
            await client.call_tool(
                "run_preprocessing",
                {
                    "function": "qc_metrics",
                    "edata_id": edata_id,
                    "params": {"qc_vars": []},
                },
            )
        ).structured_content
        assert qc["status"] == "ok"
        print("run_preprocessing qc_metrics: ok")

        obs = (
            await client.call_tool(
                "run_get",
                {
                    "function": "obs_df",
                    "edata_id": edata_id,
                    "params": {"keys": ["age"]},
                },
            )
        ).structured_content
        assert obs["status"] == "ok"
        print("run_get obs_df: ok")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patients.csv"
            path.write_text("patient_id,age,sex\np1,45,F\np2,62,M\n", encoding="utf-8")
            ing = (await client.call_tool("ingest_dataset", {"file_path": str(path)})).structured_content
            snap = (await client.call_tool("get_edata_snapshot", {"edata_id": ing["edata_id"]})).structured_content
            assert snap["n_obs"] == 2
            print("ingest_dataset + snapshot: ok")

    print("ALL_OK")


if __name__ == "__main__":
    asyncio.run(main())
