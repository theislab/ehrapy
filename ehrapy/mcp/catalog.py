"""Catalog of dispatchable ehrapy / ehrdata functions."""

from __future__ import annotations

import inspect
import re
import typing
from typing import TYPE_CHECKING, Any

import docstring_parser
import ehrdata.dt as ed_dt
import ehrdata.io as ed_io

import ehrapy as ep

if TYPE_CHECKING:
    from collections.abc import Callable

# scanpy-derived tools not listed in ep.tools.__all__
_EXTRA_TOOLS = ("leiden", "dendrogram", "dpt", "paga", "ingest")

# classes exported in __all__ but not invokable as functions
_SKIP_TOOLS = frozenset({"CausalEstimate", "CohortTracker", "Literal"})

# plot module imports many names; skip non-functions
_SKIP_PLOT = frozenset({"Colormaps", "LinearSegmentedColormap", "hv"})


def _plot_functions() -> list[str]:
    names: list[str] = []
    for name in dir(ep.plot):
        if name.startswith("_") or name in _SKIP_PLOT:
            continue
        obj = getattr(ep.plot, name)
        if inspect.isfunction(obj):
            names.append(name)
    return sorted(names)


def _tools_functions() -> list[str]:
    names = [n for n in ep.tools.__all__ if n not in _SKIP_TOOLS]
    for extra in _EXTRA_TOOLS:
        if extra not in names and hasattr(ep.tools, extra):
            names.append(extra)
    return sorted(names)


NAMESPACES: dict[str, dict[str, Any]] = {
    "preprocessing": {
        "module": ep.preprocessing,
        "functions": list(ep.preprocessing.__all__),
        "kind": "edata",
        "description": "Quality control, encoding, imputation, normalization, filtering, PCA, neighbors (ep.pp.*)",
    },
    "analysis": {
        "module": ep.tools,
        "functions": _tools_functions(),
        "kind": "edata",
        "description": "Analysis: survival, causal, embedding, clustering, feature ranking (ep.tl.*)",
    },
    "get": {
        "module": ep.get,
        "functions": list(ep.get.__all__),
        "kind": "get",
        "description": "Read obs/var tables and ranked feature results (ep.get.*)",
    },
    "plot": {
        "module": ep.plot,
        "functions": _plot_functions(),
        "kind": "plot",
        "description": "Visualization; renders PNG plots (ep.pl.*)",
    },
    "io": {
        "module": ed_io,
        "functions": [
            "read_csv",
            "read_h5ad",
            "read_h5ed",
            "read_zarr",
            "from_pandas",
            "write_h5ad",
            "write_h5ed",
            "write_zarr",
            "to_pandas",
        ],
        "kind": "io",
        "description": "Load/save EHRData from files (ehrdata.io.*)",
    },
    "demo": {
        "module": ed_dt,
        "functions": list(ed_dt.__all__),
        "kind": "demo",
        "description": "Built-in demo cohorts (ehrdata.dt.*)",
    },
}

_NAMESPACE_ALIASES = {
    "tools": "analysis",
    "dt": "demo",
}


def _canonical_namespace(namespace: str) -> str:
    return _NAMESPACE_ALIASES.get(namespace, namespace)


def list_namespaces() -> list[str]:
    """Return sorted dispatch namespace names."""
    return sorted(NAMESPACES)


def list_functions(namespace: str) -> list[str]:
    """Return function names in a dispatch namespace."""
    canonical = _canonical_namespace(namespace)
    spec = NAMESPACES.get(canonical)
    if spec is None:
        raise KeyError(namespace)
    return list(spec["functions"])


def get_callable(namespace: str, function: str) -> Callable[..., Any]:
    """Resolve a namespaced ehrapy/ehrdata callable."""
    canonical = _canonical_namespace(namespace)
    spec = NAMESPACES.get(canonical)
    if spec is None:
        raise KeyError(f"Unknown namespace '{namespace}'")
    if function not in spec["functions"]:
        raise KeyError(f"Unknown function '{function}' in namespace '{namespace}'")
    return getattr(spec["module"], function)


def get_namespace_kind(namespace: str) -> str:
    """Return the dispatch kind for a namespace."""
    canonical = _canonical_namespace(namespace)
    return NAMESPACES[canonical]["kind"]


def _literal_choices(ann: Any) -> list[str] | None:
    """Return string choices if ``ann`` is (or wraps, e.g. via Optional/Union) a typing.Literal."""
    if typing.get_origin(ann) is typing.Literal:
        return [str(a) for a in typing.get_args(ann)]
    if hasattr(ann, "__args__"):
        for arg in ann.__args__:
            if typing.get_origin(arg) is typing.Literal:
                return [str(a) for a in typing.get_args(arg)]
    return None


_CHOICE_LIKE_PARAM_NAMES = frozenset({"backend", "mode", "method", "metric", "flavor", "strategy", "how", "which"})
_CHOICE_STOPWORDS = frozenset({"None", "True", "False", "strategy"})


def _looks_like_choice_param(param_name: str, desc: str) -> bool:
    return param_name in _CHOICE_LIKE_PARAM_NAMES or "choose from" in desc.lower() or "one of" in desc.lower()


def _quoted_choices(desc: str) -> list[str]:
    """Extract unique quoted tokens from a docstring description, dropping known stopwords."""
    seen: set[str] = set()
    res: list[str] = []
    for q in re.findall(r"['\"]([a-zA-Z0-9_\-]+)['\"]", desc):
        if q not in seen and q not in _CHOICE_STOPWORDS:
            seen.add(q)
            res.append(q)
    return res


def _extract_choices(param_name: str, ann: Any, desc: str | None) -> list[str] | None:
    """Infer valid choices for a parameter from its Literal annotation or docstring text."""
    literal_choices = _literal_choices(ann)
    if literal_choices is not None:
        return literal_choices
    if not desc or not _looks_like_choice_param(param_name, desc):
        return None
    return _quoted_choices(desc) or None


def _synthesize_example_call(namespace: str, function: str, params: list[dict[str, Any]]) -> str:
    canonical = _canonical_namespace(namespace)
    if canonical == "demo":
        return f"load_demo_dataset(dataset='{function}')"

    tool_map = {
        "preprocessing": "run_preprocessing",
        "analysis": "run_analysis",
        "get": "run_get",
        "plot": "run_plot",
        "io": "run_io",
    }
    tool_name = tool_map.get(canonical, "run_preprocessing")

    sample_params: dict[str, Any] = {}
    for p in params:
        name = p["name"]
        if name in ("edata", "copy", "kwargs", "inplace", "in_place"):
            continue
        if p.get("choices"):
            sample_params[name] = p["choices"][0]
        elif name in ("n_neighbours", "n_neighbors"):
            sample_params[name] = 5
        elif name == "keys":
            sample_params[name] = ["age"]
        elif name == "groupby":
            sample_params[name] = "leiden"
        elif name == "qc_vars":
            sample_params[name] = []
        elif len(sample_params) < 1 and "default" not in p:
            sample_params[name] = "..."
        if len(sample_params) >= 2:
            break

    if sample_params:
        params_str = ", ".join(f"'{k}': {repr(v) if v != '...' else '...'}" for k, v in sample_params.items())
        return f"{tool_name}(function='{function}', params={{{params_str}}})"
    return f"{tool_name}(function='{function}')"


def _param_info_entry(name: str, param: inspect.Parameter, doc_params: dict[str, str | None]) -> dict[str, Any]:
    """Build the help-payload entry for a single function parameter."""
    entry: dict[str, Any] = {"name": name}
    if param.default is not inspect.Parameter.empty:
        entry["default"] = repr(param.default)
    if param.annotation is not inspect.Parameter.empty:
        entry["annotation"] = str(param.annotation)

    desc = doc_params.get(name) or ""
    # Clean up whitespace in description
    desc_clean = " ".join(desc.split())
    if desc_clean:
        entry["description"] = desc_clean

    choices = _extract_choices(name, param.annotation, desc)
    if choices:
        entry["choices"] = choices

    return entry


def _first_example_snippet(parsed: Any) -> str:
    """Return the first non-empty example snippet from a parsed docstring, or ''."""
    for ex in parsed.examples:
        snippet = (ex.snippet or ex.description or "").strip()
        if snippet:
            return snippet[:600]
    return ""


def function_help(namespace: str, function: str) -> dict[str, Any]:
    """Return signature metadata, numpydoc parameter descriptions, choices, and example call."""
    canonical = _canonical_namespace(namespace)
    fn = get_callable(canonical, function)
    sig = inspect.signature(fn)
    raw_doc = inspect.getdoc(fn) or ""

    parsed = docstring_parser.parse(raw_doc)
    doc_params = {p.arg_name: p.description for p in parsed.params if p.arg_name}

    params_info = [_param_info_entry(name, param, doc_params) for name, param in sig.parameters.items()]

    example_call = _synthesize_example_call(canonical, function, params_info)
    short_doc = (parsed.short_description or raw_doc.split("\n\n")[0])[:400]

    return {
        "namespace": canonical,
        "function": function,
        "kind": get_namespace_kind(canonical),
        "parameters": params_info,
        "short_description": short_doc,
        "example_snippet": _first_example_snippet(parsed),
        "example_call": example_call,
    }


def function_help_markdown(namespace: str, function: str) -> str:
    """Render function help as compact Markdown under 2,500 chars."""
    info = function_help(namespace, function)
    canonical = info["namespace"]
    fn_name = info["function"]

    lines: list[str] = []
    lines.append(f"### `{canonical}.{fn_name}`")
    if info.get("short_description"):
        lines.append(f"{info['short_description']}\n")

    lines.append("| Parameter | Type | Default | Choices | Description |")
    lines.append("|---|---|---|---|---|")

    for p in info["parameters"]:
        p_name = p["name"]
        p_type = p.get("annotation", "—")
        if len(p_type) > 25:
            p_type = p_type[:22] + "..."
        p_default = p.get("default", "—")
        p_choices = ", ".join(p["choices"]) if p.get("choices") else "—"
        p_desc = p.get("description", "")
        if len(p_desc) > 80:
            p_desc = p_desc[:77] + "..."
        # escape pipes
        p_desc = p_desc.replace("|", "\\|")
        lines.append(f"| `{p_name}` | {p_type} | {p_default} | {p_choices} | {p_desc} |")

    if info.get("example_snippet"):
        lines.append("\n**Example:**")
        lines.append(f"```python\n{info['example_snippet']}\n```")

    lines.append(f"\n**Example call:**\n`{info['example_call']}`")

    md = "\n".join(lines)
    if len(md) > 2500:
        # Truncate descriptions if needed
        md = md[:2450] + "\n... [Help truncated]"
    return md


def catalog_summary() -> dict[str, Any]:
    """Summarize every dispatch namespace and its functions."""
    return {
        ns: {
            "kind": spec["kind"],
            "count": len(spec["functions"]),
            "description": spec["description"],
            "functions": spec["functions"],
        }
        for ns, spec in NAMESPACES.items()
    }
