"""Schema portability floor tests (T13).

Ensures all tool inputSchemas comply with strict LLM client schema requirements (no $ref, valid primitive types).
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import Client

from ehrapy.mcp.server import mcp

_ALLOWED_TYPES = {"string", "number", "integer", "boolean", "object", "array", "null"}


def _run(coro):
    return asyncio.run(coro)


def _validate_schema_node(node: Any, path: str = "") -> None:
    if not isinstance(node, dict):
        return

    # No $ref or definitions
    assert "$ref" not in node, f"$ref found at {path}"
    assert "definitions" not in node, f"definitions found at {path}"
    assert "$defs" not in node, f"$defs found at {path}"

    if "type" in node:
        t = node["type"]
        if isinstance(t, str):
            assert t in _ALLOWED_TYPES, f"Invalid type '{t}' at {path}"
        elif isinstance(t, list):
            for sub_t in t:
                assert sub_t in _ALLOWED_TYPES, f"Invalid type '{sub_t}' in type union at {path}"

    if "properties" in node and isinstance(node["properties"], dict):
        for prop_name, prop_val in node["properties"].items():
            _validate_schema_node(prop_val, f"{path}.properties.{prop_name}")

    if "items" in node:
        _validate_schema_node(node["items"], f"{path}.items")


def test_tool_schemas_compliance() -> None:
    """Validate that every tool's inputSchema satisfies the Gemini-safe floor."""

    async def _test():
        async with Client(mcp) as client:
            tools = await client.list_tools()
            assert len(tools) == 14
            for t in tools:
                schema = t.inputSchema
                assert isinstance(schema, dict), f"Tool {t.name} inputSchema is not a dict"
                assert schema.get("type") == "object", f"Tool {t.name} schema root type is not 'object'"
                _validate_schema_node(schema, path=t.name)

    _run(_test())
