"""Tests for LRU cache, read-only guarantees, filesystem confinement, and TTL purge (T1, T4, T14, T15)."""

from __future__ import annotations

import asyncio
import os
import stat
import tempfile
from typing import TYPE_CHECKING

import ehrdata.dt as dt
import pytest
from fastmcp import Client

from ehrapy.mcp.edata_store import _MEMORY_CACHE, load_edata, persist_edata, save_edata
from ehrapy.mcp.policy import (
    PathNotAllowedError,
    ReadOnlyModeError,
    check_path_allowed,
    is_read_only_mode,
)
from ehrapy.mcp.registry import MCPRegistry, registry
from ehrapy.mcp.server import mcp

if TYPE_CHECKING:
    from pathlib import Path


def _run(coro):
    return asyncio.run(coro)


def test_read_only_get_does_not_touch_disk() -> None:
    """Test that run_get and get_edata_snapshot do not modify dataset mtime on disk (T1)."""

    async def _test():
        async with Client(mcp) as client:
            demo_res = await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
            edata_id = demo_res.structured_content["edata_id"]

            file_path = registry.cache_dir() / "edata" / f"{edata_id}.h5ed"
            assert file_path.exists()
            mtime_before = file_path.stat().st_mtime_ns

            # Call run_get
            get_res = await client.call_tool(
                "run_get", {"function": "obs_df", "edata_id": edata_id, "params": {"keys": ["age"]}}
            )
            assert get_res.structured_content["status"] == "ok"
            mtime_after_get = file_path.stat().st_mtime_ns
            assert mtime_after_get == mtime_before

            # Call get_edata_snapshot
            snap_res = await client.call_tool("get_edata_snapshot", {"edata_id": edata_id})
            assert snap_res.structured_content["status"] == "ok"
            mtime_after_snap = file_path.stat().st_mtime_ns
            assert mtime_after_snap == mtime_before

    _run(_test())


def test_lru_cache_hit_and_eviction() -> None:
    """Test that LRU cache keeps entries in memory and evicts oldest upon exceeding capacity (T4)."""
    _MEMORY_CACHE.clear()
    assert len(_MEMORY_CACHE) == 0

    # Load demo dataset
    edata1 = dt.mimic_2()
    rec1 = save_edata(edata1, name="edata1")
    assert rec1.edata_id in _MEMORY_CACHE

    # Access via load_edata (hit)
    loaded1 = load_edata(rec1.edata_id)
    assert loaded1.n_obs == edata1.n_obs

    # Fill cache beyond capacity (capacity = 3)
    edata2 = dt.mimic_2()
    save_edata(edata2, name="edata2")
    edata3 = dt.mimic_2()
    save_edata(edata3, name="edata3")
    edata4 = dt.mimic_2()
    save_edata(edata4, name="edata4")

    # Ensure cache length <= 3
    assert len(_MEMORY_CACHE) <= 3
    # rec1 should have been evicted from in-memory cache, but remains on disk
    assert rec1.edata_id not in _MEMORY_CACHE
    # Can still reload from disk
    reloaded1 = load_edata(rec1.edata_id)
    assert reloaded1.n_obs == edata1.n_obs
    assert rec1.edata_id in _MEMORY_CACHE


def test_filesystem_confinement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that allowed roots restricts path access (T14)."""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    forbidden_dir = tmp_path / "forbidden"
    forbidden_dir.mkdir()

    monkeypatch.setenv("EHRAPY_MCP_ALLOWED_ROOTS", str(allowed_dir))

    # Path inside allowed dir
    valid_file = allowed_dir / "data.csv"
    valid_file.write_text("a,b\n1,2\n")
    checked = check_path_allowed(valid_file, for_write=False)
    assert checked.resolve() == valid_file.resolve()

    # Path outside allowed dir
    invalid_file = forbidden_dir / "secret.csv"
    invalid_file.write_text("a,b\n1,2\n")
    with pytest.raises(PathNotAllowedError):
        check_path_allowed(invalid_file, for_write=False)


def test_read_only_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that EHRAPY_MCP_READ_ONLY disables write operations (T14)."""
    monkeypatch.setenv("EHRAPY_MCP_READ_ONLY", "1")
    assert is_read_only_mode() is True

    target = tmp_path / "out.csv"
    with pytest.raises(ReadOnlyModeError):
        check_path_allowed(target, for_write=True)


def test_cache_directory_permissions_and_purge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test cache 0o700 permissions and TTL purge (T15)."""
    custom_cache = tmp_path / "mcp_cache"
    monkeypatch.setenv("EHRAPY_MCP_CACHE_DIR", str(custom_cache))

    reg = MCPRegistry()
    cache_path = reg.cache_dir()
    assert cache_path.exists()
    mode = stat.S_IMODE(cache_path.stat().st_mode)
    assert mode & 0o700 == 0o700

    # Test purge runs without errors
    purged = reg.purge(older_than_days=7)
    assert isinstance(purged, int)
