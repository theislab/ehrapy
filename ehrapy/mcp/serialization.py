"""SOTA 3-tier serialization for ehrapy return values.

Markdown-first for human/agent reasoning, compact typed JSON metadata for structuredContent.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from ehrdata import EHRData
from fastmcp.utilities.types import Image

if TYPE_CHECKING:
    from pathlib import Path

    pass

# Budget constants
_DEFAULT_MAX_CHARS = 10_000
_TIER1_MAX_CHARS = 2_500
_TIER3_DEFAULT_ROWS = 20
_MAX_REPR_CHARS = 1_000

# Information ranking weights
W_RELEVANCE = 2.0
W_MISSING = 1.0
W_DISPERSION = 1.0


def _get_max_result_chars() -> int:
    env_val = os.environ.get("EHRAPY_MCP_MAX_RESULT_CHARS", "").strip()
    if env_val:
        try:
            return max(500, int(env_val))
        except ValueError:
            pass
    return _DEFAULT_MAX_CHARS


def _format_cell(val: Any) -> str:
    if val is None:
        return "NaN"
    if isinstance(val, (list, tuple, set, dict, np.ndarray)):
        # pd.isna on a sequence returns an array, whose truth value is ambiguous.
        return str(val)[:40]
    if isinstance(val, float) and math.isnan(val):
        return "NaN"
    try:
        if pd.isna(val):
            return "NaN"
    except (TypeError, ValueError):
        pass
    if isinstance(val, (float, np.floating)):
        return f"{val:.4g}"
    s = str(val).replace("\n", " ").replace("|", "\\|")
    return s if len(s) <= 40 else s[:37] + "..."


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown pipe table."""
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(_format_cell(val) for val in row) + " |"
        lines.append(row_str)
    return "\n".join(lines)


def _compute_entropy(series: pd.Series) -> float:
    """Compute normalized Shannon entropy in [0, 1]."""
    counts = series.dropna().value_counts(normalize=True)
    k = len(counts)
    if k <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in counts if p > 0)
    max_h = math.log(k)
    return (h / max_h) if max_h > 0 else 0.0


def _compute_dispersion(series: pd.Series) -> float:
    """Compute normalized dispersion metric in [0, 1]."""
    valid = series.dropna()
    if len(valid) <= 1:
        return 0.0
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        v_min = valid.min()
        v_max = valid.max()
        rng = v_max - v_min
        if rng == 0:
            return 0.0
        scaled = (valid - v_min) / rng
        return float(min(1.0, scaled.std()))
    return _compute_entropy(series)


def _extract_relevant_columns(params: dict[str, Any] | None) -> set[str]:
    """Extract column names mentioned in call parameters."""
    if not params:
        return set()
    relevant: set[str] = set()
    for key in (
        "keys",
        "groupby",
        "duration_col",
        "event_col",
        "target",
        "covariates",
        "qc_vars",
        "color",
        "layer",
    ):
        val = params.get(key)
        if isinstance(val, str):
            relevant.add(val)
        elif isinstance(val, (list, tuple, set)):
            relevant.update(str(x) for x in val)
    return relevant


def _rank_columns(df: pd.DataFrame, relevant_cols: set[str]) -> list[tuple[str, float]]:
    """Rank columns by information score: w_m*missing + w_v*dispersion + w_r*relevance."""
    scored: list[tuple[str, float]] = []
    for col in df.columns:
        col_str = str(col)
        missing_frac = float(df[col].isna().mean())
        disp = _compute_dispersion(df[col])
        rel = 1.0 if col_str in relevant_cols else 0.0
        score = (W_MISSING * missing_frac) + (W_DISPERSION * disp) + (W_RELEVANCE * rel)
        scored.append((col_str, score))
    # Deterministic tie-break by column name
    scored.sort(key=lambda item: (-item[1], str(item[0])))
    return scored


def _summarize_column(series: pd.Series) -> str:
    """Produce summary string for a single column profile."""
    valid = series.dropna()
    if len(valid) == 0:
        return "all missing"

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        mean_val = valid.mean()
        std_val = valid.std() if len(valid) > 1 else 0.0
        min_val = valid.min()
        max_val = valid.max()
        return f"{mean_val:.2f} ± {std_val:.2f} [{min_val:.2f}, {max_val:.2f}]"

    if pd.api.types.is_datetime64_any_dtype(series):
        return f"{valid.min()} → {valid.max()}"

    top_counts = valid.value_counts().head(3)
    parts = [f"{k}: {v}" for k, v in top_counts.items()]
    return ", ".join(parts)


def _profile_dataframe(
    df: pd.DataFrame,
    relevant_cols: set[str],
    *,
    max_profile_cols: int = 40,
) -> tuple[str, dict[str, Any]]:
    """Render Tier 2 column profile table."""
    total_cols = df.shape[1]
    ranked = _rank_columns(df, relevant_cols)

    selected_cols = [c for c, _ in ranked[:max_profile_cols]]
    dropped_count = total_cols - len(selected_cols)

    headers = ["column", "dtype", "non-null %", "unique", "summary"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for col_name in selected_cols:
        s = df[col_name]
        dtype_str = str(s.dtype)
        non_null_pct = f"{s.notna().mean() * 100:.1f}%"
        nunique_str = str(s.nunique())
        summ = _summarize_column(s).replace("|", "\\|")
        row = f"| `{col_name}` | {dtype_str} | {non_null_pct} | {nunique_str} | {summ} |"
        lines.append(row)

    md = "\n".join(lines)
    steering_notes = []
    if dropped_count > 0:
        steering_notes.append(
            f"\n\n*Showing profile for {len(selected_cols)} of {total_cols} columns (ranked by missingness and variance). {dropped_count} columns omitted.*"
        )

    full_md = md + "".join(steering_notes)
    meta = {
        "type": "dataframe_profile",
        "shape": list(df.shape),
        "total_columns": total_cols,
        "profiled_columns": len(selected_cols),
        "dropped_columns": dropped_count,
    }
    return full_md, meta


def _find_strat_col(df: pd.DataFrame, relevant_cols: set[str]) -> str | None:
    """Find a relevant, non-numeric column to stratify the middle sample by."""
    for col in relevant_cols:
        if col in df.columns and (not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])):
            return col
    return None


def _middle_sample_positions(mid_df: pd.DataFrame, strat_col: str | None, middle_n: int) -> list[int]:
    """Pick ``middle_n`` row positions (relative to ``mid_df``), stratified when useful."""
    pos_series = pd.Series(range(len(mid_df)))
    if strat_col is None or mid_df[strat_col].nunique() <= 1:
        return list(pos_series.sample(min(middle_n, len(pos_series)), random_state=42))
    try:
        strat_labels = pd.Series(mid_df[strat_col].to_numpy())
        grouped = pos_series.groupby(strat_labels, group_keys=False)
        sampled_rel = grouped.apply(
            lambda g: g.sample(max(1, int(len(g) / len(pos_series) * middle_n)), random_state=42)
        )
        return list(sampled_rel.iloc[:middle_n])
    except Exception:  # noqa: BLE001
        return list(pos_series.sample(min(middle_n, len(pos_series)), random_state=42))


def _sample_rows(
    df: pd.DataFrame,
    relevant_cols: set[str],
    *,
    row_budget: int = _TIER3_DEFAULT_ROWS,
) -> pd.DataFrame:
    """Sample head 5 + tail 5 + stratified/uniform sample to row_budget with fixed seed 42."""
    n_rows = len(df)
    if n_rows <= row_budget:
        result = df.copy()
        result.insert(0, "sample", "all")
        return result

    head_n = min(5, n_rows)
    tail_n = min(5, n_rows - head_n)
    middle_n = max(0, row_budget - head_n - tail_n)

    head_pos = list(range(head_n))
    tail_pos = list(range(n_rows - tail_n, n_rows)) if tail_n > 0 else []
    mid_pos_range = list(range(head_n, n_rows - tail_n))

    sample_pos: list[int] = []
    if middle_n > 0 and mid_pos_range:
        mid_df = df.iloc[mid_pos_range]
        strat_col = _find_strat_col(df, relevant_cols)
        chosen_rel = _middle_sample_positions(mid_df, strat_col, middle_n)
        sample_pos = [mid_pos_range[p] for p in chosen_rel]

    all_positions = head_pos + sample_pos + tail_pos
    sample_tags = ["head"] * len(head_pos) + ["sample"] * len(sample_pos) + ["tail"] * len(tail_pos)

    sampled_df = df.iloc[all_positions].copy()
    sampled_df.insert(0, "sample", sample_tags)
    return sampled_df


def _serialize_empty_dataframe(
    df: pd.DataFrame, params: dict[str, Any], function: str, edata: EHRData | None
) -> tuple[dict[str, Any], str]:
    """Serialize a DataFrame with no rows or no columns, steering toward a `keys` param if applicable."""
    if not params.get("keys") and function in ("obs_df", "var_df", "rank_features_groups_df"):
        avail_cols: list[str] = []
        if edata is not None:
            if function == "obs_df":
                avail_cols = list(edata.obs.columns[:30])
            elif function == "var_df":
                avail_cols = list(edata.var.columns[:30])
        avail_str = f" Available columns: {', '.join(avail_cols)}" if avail_cols else ""
        meta = {
            "status": "ok_empty",
            "shape": list(df.shape),
            "agent_action": f"No columns requested. Specify columns using `params={{'keys': [...]}}`.{avail_str}",
        }
        md = f"# Result: ok_empty\n\nReturned table shape is `{list(df.shape)}`.\n\n**Action:** {meta['agent_action']}"
        return meta, md
    meta = {"type": "dataframe", "shape": list(df.shape)}
    return meta, f"Empty DataFrame (shape: {list(df.shape)})."


def _tier3_sampled_view(
    df: pd.DataFrame,
    relevant_cols: set[str],
    response_format: Literal["concise", "detailed"],
    params: dict[str, Any],
) -> tuple[dict[str, Any], str] | None:
    """Return a Tier-3 sampled-row view when rows were explicitly requested, else None."""
    if response_format != "detailed" and "rows" not in params:
        return None
    sampled = _sample_rows(df, relevant_cols)
    table_md = _df_to_markdown_table(sampled)
    meta: dict[str, Any] = {"type": "dataframe", "shape": list(df.shape), "tier": 3, "sampled_rows": len(sampled)}
    md = f"### DataFrame ({df.shape[0]} rows × {df.shape[1]} columns — sampled view)\n\n" + table_md
    return meta, md


def _tier1_full_table(df: pd.DataFrame) -> tuple[dict[str, Any], str] | None:
    """Return the whole table rendered as Markdown if it fits the Tier-1 budget, else None."""
    # Bound the work first -- rendering a 100k-row frame to Markdown just to discover it is
    # too big is the exact overflow this tier is meant to prevent.
    cells = df.shape[0] * max(1, df.shape[1])
    if cells > _TIER1_MAX_CHARS:
        return None
    full_table_md = _df_to_markdown_table(df)
    if len(full_table_md) > _TIER1_MAX_CHARS:
        return None
    meta: dict[str, Any] = {"type": "dataframe", "shape": list(df.shape), "tier": 1}
    md = f"### DataFrame ({df.shape[0]} rows × {df.shape[1]} columns)\n\n" + full_table_md
    return meta, md


def _tier2_profile(df: pd.DataFrame, relevant_cols: set[str], is_whole_table: bool) -> tuple[dict[str, Any], str]:
    """Render the Tier-2 column-profile view (default for large or whole-table results)."""
    profile_md, profile_meta = _profile_dataframe(df, relevant_cols)
    profile_meta["tier"] = 2
    header = f"### Column Profile ({df.shape[0]} rows × {df.shape[1]} columns)\n\n"
    steering = ""
    if is_whole_table:
        steering = "\n\n*Full table not returned — use `export_edata` to write it to disk, or `run_get` with `keys=[...]` for specific columns.*"
    return profile_meta, header + profile_md + steering


def _serialize_dataframe(
    df: pd.DataFrame,
    *,
    response_format: Literal["concise", "detailed"] = "concise",
    params: dict[str, Any] | None = None,
    function: str = "",
    edata: EHRData | None = None,
) -> tuple[dict[str, Any], str]:
    """Serialize DataFrame according to Tier 1, 2, or 3 rules."""
    params = params or {}
    relevant_cols = _extract_relevant_columns(params)

    if df.shape[0] == 0 or df.shape[1] == 0:
        return _serialize_empty_dataframe(df, params, function, edata)

    is_whole_table = function == "to_pandas"

    if not is_whole_table:
        tier3 = _tier3_sampled_view(df, relevant_cols, response_format, params)
        if tier3 is not None:
            return tier3
        tier1 = _tier1_full_table(df)
        if tier1 is not None:
            return tier1

    return _tier2_profile(df, relevant_cols, is_whole_table)


def _unique_plot_path(plots_dir: Path, stem: str) -> Path:
    plots_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = plots_dir / f"{stem}.png"
    counter = 0
    while path.exists():
        counter += 1
        path = plots_dir / f"{stem}_{counter}.png"
    return path


_MAX_PLOT_BYTES = 800 * 1024
_DPI_LADDER = (100, 72, 56, 40)


def _save_figure(fig: Any, plots_dir: Path, stem: str) -> tuple[dict[str, Any], Path]:
    """Save a figure, stepping the DPI down until it fits the image budget."""
    path = _unique_plot_path(plots_dir, stem)
    size = 0
    dpi = _DPI_LADDER[0]
    for dpi in _DPI_LADDER:
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        size = path.stat().st_size
        if size <= _MAX_PLOT_BYTES:
            break
    meta: dict[str, Any] = {
        "type": "figure",
        "plot_path": str(path),
        "media_type": "image/png",
        "dpi": dpi,
        "size_bytes": size,
    }
    if size > _MAX_PLOT_BYTES:
        # Report rather than silently returning an oversized payload.
        meta["oversized"] = True
    return meta, path


# Tried in order for holoviews PNG export. The backend is named per-call rather than set
# via hv.extension(), which would mutate global state for every other plot in the process.
# bokeh is the holoviews default but its PNG export needs selenium and a headless browser.
_HOLOVIEWS_BACKENDS = ("matplotlib", "bokeh")


def _try_save_holoviews(obj: Any, plots_dir: Path) -> tuple[dict[str, Any], Path] | None:
    """Render a holoviews object (e.g. ep.pl.kaplan_meier's Overlay) to PNG."""
    try:
        import holoviews as hv
    except ImportError:
        return None

    if not isinstance(obj, hv.core.dimension.Dimensioned):
        return None

    path = _unique_plot_path(plots_dir, "holoviews")
    failures: list[str] = []
    for backend in _HOLOVIEWS_BACKENDS:
        if backend not in hv.Store.renderers:
            continue
        try:
            hv.save(obj, str(path), fmt="png", backend=backend)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{backend}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        return {
            "type": "figure",
            "plot_path": str(path),
            "media_type": "image/png",
            "render_backend": backend,
        }, path

    # Every backend failed. Report it rather than returning a success with no image.
    if path.exists():
        path.unlink(missing_ok=True)
    raise ValueError(
        "Could not export this figure to PNG with any available holoviews backend. "
        + " | ".join(failures)
        + ". Install the 'selenium' package to enable bokeh image export, or request a "
        "different plot for this result."
    )


def _try_save_visual(obj: Any, plots_dir: Path | None) -> tuple[dict[str, Any], Path] | None:
    if plots_dir is None:
        return None
    fig = getattr(obj, "figure", None)
    if fig is not None:
        return _save_figure(fig, plots_dir, stem=type(obj).__name__)
    if type(obj).__module__.startswith("matplotlib"):
        return _save_figure(obj, plots_dir, stem="matplotlib")
    return _try_save_holoviews(obj, plots_dir)


_JSON_SAFE_SCALARS = (str, bool, int, float)
_MAX_INLINE_SEQUENCE = 32


def _json_safe_ndarray(value: np.ndarray, depth: int) -> Any:
    if value.ndim == 1 and value.size <= _MAX_INLINE_SEQUENCE:
        return [_json_safe(v, depth) for v in value.tolist()]
    return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}


def _json_safe_sequence(seq_like: list[Any] | tuple[Any, ...] | set[Any], depth: int) -> list[Any]:
    seq = list(seq_like)
    out = [_json_safe(v, depth) for v in seq[:_MAX_INLINE_SEQUENCE]]
    if len(seq) > _MAX_INLINE_SEQUENCE:
        out.append(f"... ({len(seq) - _MAX_INLINE_SEQUENCE} more)")
    return out


def _json_safe_scalar(value: Any) -> Any:
    """Coerce a scalar to JSON-safe, mapping non-finite floats (NaN/inf) to None."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_safe_pandas(value: Any) -> dict[str, Any] | None:
    """Summarize a pandas Series/DataFrame, or None if ``value`` is neither."""
    if isinstance(value, pd.Series):
        return {"type": "series", "length": int(len(value)), "dtype": str(value.dtype)}
    if isinstance(value, pd.DataFrame):
        return {"type": "dataframe", "shape": list(value.shape)}
    return None


def _json_safe(value: Any, _depth: int = 0) -> Any:
    """Coerce a value into something FastMCP can place in ``structured_content``.

    structured_content is JSON-serialized by the MCP layer, so any ndarray, Series, or
    exotic object that reaches it aborts the whole tool call. Analysis results such as
    ``CausalEstimate`` carry ndarrays inside nested ``params`` dicts, so this recurses.
    """
    if value is None or isinstance(value, _JSON_SAFE_SCALARS):
        return _json_safe_scalar(value)
    if _depth >= 4:
        return f"<{type(value).__name__}>"
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _json_safe(value.item(), _depth + 1)
    if isinstance(value, np.ndarray):
        return _json_safe_ndarray(value, _depth + 1)
    pandas_meta = _json_safe_pandas(value)
    if pandas_meta is not None:
        return pandas_meta
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return _json_safe_sequence(value, _depth + 1)
    return str(value)[:200]


def _truncate_markdown(md: str, max_chars: int, suffix: str) -> str:
    """Truncate markdown at a line boundary near max_chars, appending an explanatory suffix."""
    if len(md) <= max_chars:
        return md
    cutoff = md[:max_chars].rfind("\n")
    cutoff = cutoff if cutoff > 0 else max_chars
    return md[:cutoff] + suffix


def _serialize_visual(result: Any, plots_dir: Path | None) -> tuple[dict[str, Any], list[Any]] | None:
    """Serialize a visual/plot result to a saved PNG, or None if ``result`` is not visual."""
    visual_saved = _try_save_visual(result, plots_dir)
    if visual_saved is None:
        return None
    meta, path = visual_saved
    md = f"Plot rendered successfully and saved to `{path}`."
    return meta, [md, Image(path=str(path))]


def _serialize_ehrdata(result: EHRData) -> tuple[dict[str, Any], str]:
    meta = {"type": "EHRData", "n_obs": result.n_obs, "n_vars": result.n_vars}
    md = f"**EHRData dataset:** {result.n_obs} observations × {result.n_vars} variables"
    return meta, md


def _serialize_single_dataframe(
    result: pd.DataFrame,
    *,
    response_format: Literal["concise", "detailed"],
    params: dict[str, Any],
    function: str,
    edata: EHRData | None,
    max_chars: int,
) -> tuple[dict[str, Any], str]:
    meta, md = _serialize_dataframe(
        result, response_format=response_format, params=params, function=function, edata=edata
    )
    md = _truncate_markdown(
        md, max_chars, f"\n\n*... [Output truncated at {max_chars} characters. Narrow your query with parameters.]*"
    )
    return meta, md


def _is_dataframe_collection(result: Any) -> bool:
    """True if ``result`` is a tuple/list containing at least one DataFrame."""
    return isinstance(result, (tuple, list)) and any(isinstance(x, pd.DataFrame) for x in result)


def _table_label_and_missing_note(idx: int, item: pd.DataFrame) -> tuple[str, str]:
    """Determine a section label for one table in a collection, and any missingness callout."""
    label = "Observations" if idx == 0 and "missing_values_abs" in item.columns else f"Table {idx + 1}"
    note = ""
    if "missing_values_pct" in item.columns and "missing_values_abs" in item.columns and item.shape[0] < 1000:
        label = "Variables Quality Metrics"
        top_m = item.sort_values("missing_values_pct", ascending=False).head(5)
        parts = [
            f"{idx_name} ({row['missing_values_pct']:.1f}%)"
            for idx_name, row in top_m.iterrows()
            if row["missing_values_pct"] > 0
        ]
        if parts:
            note = f"\n\n**Highest-missingness variables:** {', '.join(parts)}"
    return label, note


def _serialize_dataframe_collection(
    result: tuple[Any, ...] | list[Any],
    *,
    response_format: Literal["concise", "detailed"],
    params: dict[str, Any],
    function: str,
    edata: EHRData | None,
    max_chars: int,
) -> tuple[dict[str, Any], str]:
    """Serialize a tuple/list containing one or more DataFrames (e.g. qc_metrics)."""
    table_sections: list[str] = []
    table_metas: list[dict[str, Any]] = []
    top_missing_note = ""

    for idx, item in enumerate(result):
        if not isinstance(item, pd.DataFrame):
            table_sections.append(f"### Item {idx + 1}\n\n`{repr(item)[:500]}`")
            continue
        t_meta, t_md = _serialize_dataframe(
            item, response_format=response_format, params=params, function=function, edata=edata
        )
        label, note = _table_label_and_missing_note(idx, item)
        if note:
            top_missing_note = note
        table_sections.append(f"### {label}\n\n{t_md}")
        table_metas.append(t_meta)

    full_md = "\n\n---\n\n".join(table_sections) + top_missing_note
    full_md = _truncate_markdown(full_md, max_chars, f"\n\n*... [Output truncated at {max_chars} characters.]*")

    return {"type": "table_collection", "tables": table_metas}, full_md


def _serialize_series(result: pd.Series) -> tuple[dict[str, Any], str]:
    if len(result) <= 20:
        df_s = result.reset_index()
        md = f"### Series `{result.name or 'result'}`\n\n" + _df_to_markdown_table(df_s)
    else:
        df_s = pd.DataFrame({"index": list(result.index), "value": list(result.values)})
        prof_md, _ = _profile_dataframe(df_s, set())
        md = f"### Series `{result.name or 'result'}` (Profile)\n\n" + prof_md
    meta = {"type": "series", "name": result.name, "length": len(result)}
    return meta, md


def _serialize_ndarray(result: np.ndarray) -> tuple[dict[str, Any], str]:
    meta = {"type": "ndarray", "shape": list(result.shape), "dtype": str(result.dtype)}
    md = f"**Array result:** shape `{list(result.shape)}`, dtype `{result.dtype}`"
    return meta, md


def _serialize_array_like(result: Any) -> tuple[dict[str, Any], str] | None:
    """Serialize a pandas Series or numpy ndarray, or None if ``result`` is neither."""
    if isinstance(result, pd.Series):
        return _serialize_series(result)
    if isinstance(result, np.ndarray):
        return _serialize_ndarray(result)
    return None


def _serialize_dataclass(result: Any) -> tuple[dict[str, Any], str]:
    d = asdict(result)
    meta = {"type": type(result).__name__, **d}
    md = f"### {type(result).__name__}\n\n" + "\n".join(f"- **{k}:** `{v}`" for k, v in d.items())
    return meta, md


def _serialize_dict_result(result: dict[str, Any]) -> tuple[dict[str, Any], str]:
    meta = {str(k): (v if isinstance(v, (int, float, str, bool)) else str(type(v).__name__)) for k, v in result.items()}
    md = "### Result\n\n" + "\n".join(f"- **{k}:** `{v}`" for k, v in result.items())
    return meta, md


def _serialize_structured(result: Any) -> tuple[dict[str, Any], str] | None:
    """Serialize a dataclass instance or a plain dict, or None if ``result`` is neither."""
    if is_dataclass(result) and not isinstance(result, type):
        return _serialize_dataclass(result)
    if isinstance(result, dict):
        return _serialize_dict_result(result)
    return None


def _serialize_with_summary(result: Any) -> tuple[dict[str, Any], str] | None:
    """Serialize via a summary() method (e.g. statsmodels/lifelines results), or None if unavailable."""
    summary_fn = getattr(result, "summary", None)
    if not callable(summary_fn):
        return None
    try:
        summ_text = str(summary_fn())[:_MAX_REPR_CHARS]
    except Exception:  # noqa: BLE001
        return None
    meta = {"type": type(result).__name__}
    md = f"### {type(result).__name__} Summary\n\n```\n{summ_text}\n```"
    return meta, md


def _serialize_fallback_repr(result: Any) -> tuple[dict[str, Any], str]:
    repr_str = repr(result)[:_MAX_REPR_CHARS]
    meta = {"type": type(result).__name__}
    md = f"```\n{repr_str}\n```"
    return meta, md


def _serialize_result_inner(
    result: Any,
    *,
    plots_dir: Path | None = None,
    response_format: Literal["concise", "detailed"] = "concise",
    params: dict[str, Any] | None = None,
    function: str = "",
    edata: EHRData | None = None,
) -> tuple[dict[str, Any], list[Any] | str]:
    """Convert an ehrapy return value into dual-channel structured_content and Markdown/image content."""
    params = params or {}
    max_chars = _get_max_result_chars()

    if result is None:
        return {}, "Operation completed successfully."

    visual = _serialize_visual(result, plots_dir)
    if visual is not None:
        return visual

    if isinstance(result, EHRData):
        return _serialize_ehrdata(result)

    if isinstance(result, pd.DataFrame):
        return _serialize_single_dataframe(
            result,
            response_format=response_format,
            params=params,
            function=function,
            edata=edata,
            max_chars=max_chars,
        )

    if _is_dataframe_collection(result):
        return _serialize_dataframe_collection(
            result,
            response_format=response_format,
            params=params,
            function=function,
            edata=edata,
            max_chars=max_chars,
        )

    array_like = _serialize_array_like(result)
    if array_like is not None:
        return array_like

    structured = _serialize_structured(result)
    if structured is not None:
        return structured

    summary_result = _serialize_with_summary(result)
    if summary_result is not None:
        return summary_result

    return _serialize_fallback_repr(result)


def serialize_result(
    result: Any,
    *,
    plots_dir: Path | None = None,
    response_format: Literal["concise", "detailed"] = "concise",
    params: dict[str, Any] | None = None,
    function: str = "",
    edata: EHRData | None = None,
) -> tuple[dict[str, Any], list[Any] | str]:
    """Serialize an ehrapy return value, guaranteeing a JSON-safe metadata channel."""
    meta, content = _serialize_result_inner(
        result,
        plots_dir=plots_dir,
        response_format=response_format,
        params=params,
        function=function,
        edata=edata,
    )
    return _json_safe(meta), content
