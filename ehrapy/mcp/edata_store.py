"""Load and persist EHRData objects by edata_id with in-memory LRU caching."""

from __future__ import annotations

import collections
import dataclasses
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from ehrdata.io import read_h5ed, write_h5ed

from ehrapy.mcp.registry import DatasetRecord, registry

if TYPE_CHECKING:
    from ehrdata import EHRData

_DEFAULT_CACHE_ENTRIES = 3


def _get_max_cache_entries() -> int:
    env_val = os.environ.get("EHRAPY_MCP_CACHE_ENTRIES", "").strip()
    if env_val:
        try:
            return max(1, int(env_val))
        except ValueError:
            pass
    return _DEFAULT_CACHE_ENTRIES


class _EHRDataLRUCache:
    """In-memory LRU cache storing (mtime_ns, EHRData)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._cache: collections.OrderedDict[str, tuple[int, EHRData]] = collections.OrderedDict()

    def get(self, edata_id: str, mtime_ns: int) -> EHRData | None:
        if edata_id not in self._cache:
            return None
        cached_mtime, edata = self._cache[edata_id]
        if cached_mtime != mtime_ns:
            # Stale cache entry
            del self._cache[edata_id]
            return None
        self._cache.move_to_end(edata_id)
        return edata

    def put(self, edata_id: str, mtime_ns: int, edata: EHRData) -> None:
        if edata_id in self._cache:
            self._cache.move_to_end(edata_id)
        self._cache[edata_id] = (mtime_ns, edata)
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, edata_id: object) -> bool:
        return edata_id in self._cache


_MEMORY_CACHE = _EHRDataLRUCache(_get_max_cache_entries())


def clear_memory_cache() -> None:
    """Clear in-memory LRU cache (useful for testing)."""
    _MEMORY_CACHE.clear()


def _cache_path(edata_id: str) -> Path:
    return registry.cache_dir() / "edata" / f"{edata_id}.h5ed"


def save_edata(
    edata: EHRData,
    *,
    name: str,
    source_path: str | None = None,
    fmt: str = "h5ed",
    edata_id: str | None = None,
    parent_id: str | None = None,
) -> DatasetRecord:
    """Persist EHRData to the MCP cache and register a handle."""
    edata_id = edata_id or str(uuid.uuid4())
    path = _cache_path(edata_id)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_h5ed(edata, path)
    stat = path.stat()
    record = DatasetRecord(
        edata_id=edata_id,
        cache_path=str(path),
        source_path=source_path,
        name=name,
        format=fmt,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        ingested_at=time.time(),
        n_obs=edata.n_obs,
        n_vars=edata.n_vars,
        parent_id=parent_id,
    )
    registry.store_record(record)
    _MEMORY_CACHE.put(edata_id, record.mtime_ns, edata)
    return record


def load_edata(edata_id: str) -> EHRData:
    """Load cached EHRData for an ``edata_id`` with LRU cache lookup."""
    record = registry.get_dataset(edata_id)
    if record is None:
        raise KeyError(edata_id)
    cache = Path(record.cache_path)
    if not cache.is_file():
        raise FileNotFoundError(record.cache_path)

    # Check in-memory cache
    cached = _MEMORY_CACHE.get(edata_id, record.mtime_ns)
    if cached is not None:
        return cached

    # Disk read on miss / staleness
    edata = read_h5ed(cache)
    _MEMORY_CACHE.put(edata_id, record.mtime_ns, edata)
    return edata


def fork_edata(edata_id: str, *, name: str | None = None) -> DatasetRecord:
    """Copy cached EHRData to a new handle."""
    edata = load_edata(edata_id)
    parent = registry.get_dataset(edata_id)
    label = name or f"{parent.name if parent else edata_id}-fork"
    # Make a copy of the EHRData object for the fork
    edata_copy = edata.copy()
    return save_edata(edata_copy, name=label, parent_id=edata_id)


def persist_edata(edata_id: str, edata: EHRData) -> DatasetRecord:
    """Overwrite the cached EHRData for an existing handle (write-through)."""
    record = registry.get_dataset(edata_id)
    if record is None:
        return save_edata(edata, name=f"edata-{edata_id[:8]}")
    cache_path = Path(record.cache_path)
    write_h5ed(edata, cache_path)
    stat = cache_path.stat()
    updated = dataclasses.replace(
        record,
        n_obs=edata.n_obs,
        n_vars=edata.n_vars,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        ingested_at=time.time(),
    )
    registry.store_record(updated)
    _MEMORY_CACHE.put(edata_id, updated.mtime_ns, edata)
    return updated
