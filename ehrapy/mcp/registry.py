"""Persistent dataset registry for EHRData handles."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import platformdirs


def _probe_writable(path: Path) -> bool:
    """Return True if path can be created and written to."""
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        probe = path / f".probe_{os.getpid()}_{time.time_ns()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _get_default_cache_dir() -> Path:
    env_dir = os.environ.get("EHRAPY_MCP_CACHE_DIR")
    if env_dir:
        cand = Path(env_dir).expanduser().resolve()
        if _probe_writable(cand):
            return cand
    user_cache = Path(platformdirs.user_cache_dir("ehrapy-mcp")).resolve()
    if _probe_writable(user_cache):
        return user_cache
    temp_cache = Path(tempfile.gettempdir()) / "ehrapy-mcp"
    if _probe_writable(temp_cache):
        return temp_cache
    return Path(tempfile.mkdtemp(prefix="ehrapy_mcp_"))


def _get_default_demo_data_dir(cache_dir: Path) -> Path:
    for env_var in ("EHRAPY_MCP_DEMO_DATA_DIR", "EHRAPY_DEMO_DATA_DIR", "EHRAPY_DATA_DIR"):
        val = os.environ.get(env_var)
        if val:
            cand = Path(val).expanduser().resolve()
            if _probe_writable(cand):
                return cand
    cand = cache_dir / "demo_data"
    if _probe_writable(cand):
        return cand
    temp_demo = Path(tempfile.gettempdir()) / "ehrapy_data"
    if _probe_writable(temp_demo):
        return temp_demo
    return Path(tempfile.mkdtemp(prefix="ehrapy_demo_data_"))


@dataclass
class DatasetRecord:
    """Metadata for a cached EHRData handle."""

    edata_id: str
    cache_path: str
    name: str
    format: str
    size_bytes: int
    mtime_ns: int
    ingested_at: float
    source_path: str | None = None
    n_obs: int | None = None
    n_vars: int | None = None
    parent_id: str | None = None


class MCPRegistry:
    """Process-wide registry of cached EHRData handles."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir_path = cache_dir or _get_default_cache_dir()
        self._demo_data_dir_path = _get_default_demo_data_dir(self._cache_dir_path)
        self._ensure_cache_dir()
        self._process_lock = threading.RLock()
        self._configure_ehrapy_demo_paths()

    def _configure_ehrapy_demo_paths(self) -> None:
        """Point ehrdata/ehrapy/scanpy demo data loaders to the writable demo directory."""
        try:
            import ehrdata.core.constants as ed_const
            import ehrdata.dt.datasets as ed_datasets

            ed_const.DEFAULT_DATA_PATH = self._demo_data_dir_path
            ed_datasets.DEFAULT_DATA_PATH = self._demo_data_dir_path
        except Exception:  # noqa: BLE001
            pass
        try:
            import ehrapy as ep

            ep.settings.datasetdir = self._demo_data_dir_path
        except Exception:  # noqa: BLE001
            pass
        try:
            import scanpy as sc

            sc.settings.datasetdir = self._demo_data_dir_path
        except Exception:  # noqa: BLE001
            pass

    @property
    def _datasets_path(self) -> Path:
        return self._cache_dir_path / "datasets.json"

    @property
    def _lock_path(self) -> Path:
        return self._cache_dir_path / ".registry.lock"

    def _ensure_cache_dir(self) -> None:
        if not _probe_writable(self._cache_dir_path):
            self._cache_dir_path = _get_default_cache_dir()
        self._cache_dir_path.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            self._cache_dir_path.chmod(0o700)
        except OSError:
            pass
        for sub in ("edata", "plots", "demo_data"):
            sub_path = self._cache_dir_path / sub
            try:
                sub_path.mkdir(mode=0o700, parents=True, exist_ok=True)
                sub_path.chmod(0o700)
            except OSError:
                pass

    def cache_dir(self) -> Path:
        """Return the MCP cache root."""
        return self._cache_dir_path

    def plots_dir(self) -> Path:
        """Return the plot artifact directory."""
        return self._cache_dir_path / "plots"

    def demo_data_dir(self) -> Path:
        """Return the writable demo data directory."""
        return self._demo_data_dir_path

    def is_cache_writable(self) -> bool:
        """Return True if cache and demo directories are confirmed writable."""
        return _probe_writable(self._cache_dir_path) and _probe_writable(self._demo_data_dir_path)

    def ensure_demo_data_dir(self) -> Path:
        """Ensure demo data directory is configured and writable, falling back if needed."""
        if not _probe_writable(self._demo_data_dir_path):
            self._demo_data_dir_path = _get_default_demo_data_dir(self._cache_dir_path)
        self._configure_ehrapy_demo_paths()
        return self._demo_data_dir_path

    def store_record(self, record: DatasetRecord) -> DatasetRecord:
        """Insert or update a dataset record."""
        with self._locked_registry():
            datasets = self._load_datasets_unlocked()
            datasets[record.edata_id] = asdict(record)
            self._write_json(self._datasets_path, datasets)
        return record

    def get_dataset(self, edata_id: str) -> DatasetRecord | None:
        """Return a dataset record by handle, if present."""
        payload = self._load_datasets().get(edata_id)
        if payload is None:
            return None
        return DatasetRecord(**payload)

    def list_datasets(self) -> list[DatasetRecord]:
        """Return all registered dataset records."""
        return [DatasetRecord(**v) for v in self._load_datasets().values()]

    def delete_record(self, edata_id: str) -> None:
        """Remove a dataset record and its cache file if present."""
        with self._locked_registry():
            datasets = self._load_datasets_unlocked()
            if edata_id in datasets:
                rec = DatasetRecord(**datasets[edata_id])
                del datasets[edata_id]
                self._write_json(self._datasets_path, datasets)
                cache = Path(rec.cache_path)
                if cache.is_file():
                    try:
                        cache.unlink()
                    except OSError:
                        pass

    def purge(self, older_than_days: int | None = None) -> int:
        """Purge stale and orphaned records. Return number of records purged."""
        if older_than_days is None:
            ttl_env = os.environ.get("EHRAPY_MCP_CACHE_TTL_DAYS", "").strip()
            if ttl_env:
                try:
                    older_than_days = int(ttl_env)
                except ValueError:
                    older_than_days = None

        purged_count = 0
        cutoff = time.time() - (older_than_days * 86400) if older_than_days is not None else None

        with self._locked_registry():
            datasets = self._load_datasets_unlocked()
            to_delete: list[str] = []
            for edata_id, raw_record in datasets.items():
                record = DatasetRecord(**raw_record)
                cache_file = Path(record.cache_path)
                if not cache_file.is_file():
                    to_delete.append(edata_id)
                    continue
                if cutoff is not None and record.ingested_at < cutoff:
                    to_delete.append(edata_id)
                    try:
                        cache_file.unlink()
                    except OSError:
                        pass

            for edata_id in to_delete:
                del datasets[edata_id]
                purged_count += 1

            if to_delete:
                self._write_json(self._datasets_path, datasets)

        return purged_count

    def _load_datasets(self) -> dict[str, dict]:
        return self._load_datasets_unlocked()

    def _load_datasets_unlocked(self) -> dict[str, dict]:
        if not self._datasets_path.is_file():
            return {}
        try:
            return json.loads(self._datasets_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    @contextmanager
    def _locked_registry(self):
        with self._process_lock:
            self._ensure_cache_dir()
            try:
                with self._lock_path.open("a+", encoding="utf-8") as handle:
                    self._acquire_file_lock(handle)
                    try:
                        yield
                    finally:
                        self._release_file_lock(handle)
            except OSError:
                yield

    @staticmethod
    def _acquire_file_lock(handle) -> None:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass

    @staticmethod
    def _release_file_lock(handle) -> None:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError:
            pass

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)


registry = MCPRegistry()
