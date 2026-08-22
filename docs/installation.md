```{highlight} shell

```

# Installation

## Stable release

To install ehrapy, run this command in your terminal:

```console
pip install ehrapy
```

This is the preferred method to install ehrapy, as it will always install the most recent stable release.

If you don't have [pip] installed, this [Python installation guide] can guide you through the process.

If you run into "RuntimeError: CMake must be installed to build qdldl" ensure that you have CMake installed to build lightgbm.
Run `conda install -c anaconda cmake` and `conda install -c conda-forge lightgbm` to do so.

### Optional dependencies

#### leiden clustering

To use `ehrapy.tools.leiden`, install the `leiden` extra (which pulls in `igraph`):

```console
pip install ehrapy[leiden]
```

#### MCP server

To expose ehrapy to MCP clients (Cursor, Claude Desktop, Gemini CLI, etc.), install the `mcp` extra:

```console
pip install ehrapy[mcp]
```

Then start the server with `ehrapy-mcp` or `python -m ehrapy.mcp`.

##### Data at rest & Security

The ehrapy MCP server operates with safe data-handling defaults:

- **User-scoped Cache:** Datasets ingested or manipulated through MCP are persisted in a user-scoped cache directory (`platformdirs.user_cache_dir("ehrapy-mcp")`, e.g. `~/.cache/ehrapy-mcp` on Linux or `~/Library/Caches/ehrapy-mcp` on macOS).
- **Filesystem Permissions:** The cache directory tree is strictly created with POSIX `0o700` permissions (readable and writable only by the owner).
- **Plot Artifacts:** `run_plot` writes rendered PNGs to `<cache>/plots`. It is annotated `readOnlyHint: true` because it does not modify the dataset, but it is not free of side effects on disk.
- **Filesystem Confinement:** Set `EHRAPY_MCP_ALLOWED_ROOTS` (colon-separated paths) to restrict file ingestion and export to specific directory trees. Any path traversal or access outside these roots will be rejected.
- **Read-Only Mode:** Set `EHRAPY_MCP_READ_ONLY=1` to block agent-directed writes to the host filesystem — `export_edata` and `run_io(function="write_*")` are rejected with `READ_ONLY_MODE`.
  This is *not* a no-disk-writes mode: the server still writes its own working state, namely cached datasets under the MCP cache directory (including patient-derived data loaded via `load_demo_dataset` or `ingest_dataset`) and rendered PNG artifacts under `<cache>/plots`. To constrain where that state lands, set `EHRAPY_MCP_CACHE_DIR`; to guarantee no writes at all, run the server on a read-only filesystem or an ephemeral container.
- **In-Memory LRU Cache:** An LRU cache holds active dataset objects in memory (default capacity: 3 datasets, configurable via `EHRAPY_MCP_CACHE_ENTRIES`) to minimize disk I/O.
- **Automatic TTL Purge:** Datasets older than a specified threshold (e.g. 7 days) can be automatically evicted from disk using the registry purge mechanism.

| Environment Variable | Description | Default |
| --- | --- | --- |
| `EHRAPY_MCP_ALLOWED_ROOTS` | Colon-separated directory roots allowed for ingestion and export | *(unrestricted)* |
| `EHRAPY_MCP_READ_ONLY` | Set to `1` or `true` to reject agent-directed exports and `write_*` I/O (the internal dataset/plot cache is still written) | `0` |
| `EHRAPY_MCP_CACHE_DIR` | Custom path for the MCP dataset cache | Platform user cache dir |
| `EHRAPY_MCP_CACHE_ENTRIES` | Maximum number of EHRData objects kept in memory LRU | `3` |
| `EHRAPY_MCP_MAX_RESULT_CHARS` | Hard character cap for serialized Markdown results | `10000` |

## From sources

The sources for ehrapy can be downloaded from the [Github repo].

You can either clone the public repository:

```console
git clone git://github.com/theislab/ehrapy
```

Or download the [tarball]:

```console
curl -OJL https://github.com/theislab/ehrapy/tarball/master
```

[github repo]: https://github.com/theislab/ehrapy
[pip]: https://pip.pypa.io
[python installation guide]: http://docs.python-guide.org/en/latest/starting/installation/
[tarball]: https://github.com/theislab/ehrapy/tarball/master
