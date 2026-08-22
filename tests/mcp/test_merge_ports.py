"""Regression tests for fixes ported from origin/feat/add-mcp-tooling and found by dogfooding.

Each test here corresponds to a defect that was live at merge time. The mapping back to
origin's issue numbers is in the merge commit message (`git log --grep='port origin fixes'`).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re

import pytest
from fastmcp import Client

from ehrapy.mcp.catalog import get_callable
from ehrapy.mcp.errors import classify_exception_error
from ehrapy.mcp.server import mcp
from ehrapy.mcp.session import get_session, reset_sessions
from ehrapy.mcp.steering import _WORKFLOW_GRAPH

_TOOL_TO_NAMESPACE = {
    "run_preprocessing": "preprocessing",
    "run_analysis": "analysis",
    "run_get": "get",
    "run_plot": "plot",
    "run_io": "io",
}


def _run(coro):
    return asyncio.run(coro)


def _steering_suggestions():
    for key, suggestions in _WORKFLOW_GRAPH.items():
        for call, _reason in suggestions:
            match = re.match(r"(\w+)\((.*)\)$", call, re.S)
            if not match or match.group(1) not in _TOOL_TO_NAMESPACE:
                continue
            fn_match = re.search(r"function='([^']+)'", match.group(2))
            if not fn_match:
                continue
            yield key, call, _TOOL_TO_NAMESPACE[match.group(1)], fn_match.group(1), match.group(2)


def test_every_steering_target_resolves() -> None:
    """Every function the steering graph suggests must exist in the catalog."""
    for _key, _call, namespace, function, _inner in _steering_suggestions():
        get_callable(namespace, function)  # raises KeyError if the suggestion is stale


def test_steering_suggestions_name_their_required_params() -> None:
    """A suggested call must mention every required parameter of its target.

    Dogfooding found `encode`, `cox_ph`, and `covariate_balance` advertised without the
    arguments they require, so an agent following the steering hit an immediate error.
    """
    offenders = []
    for key, call, namespace, function, inner in _steering_suggestions():
        signature = inspect.signature(get_callable(namespace, function))
        required = [
            p.name
            for p in signature.parameters.values()
            if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ][1:]  # skip the edata/adata parameter, which dispatch binds
        missing = [r for r in required if r not in inner]
        if missing:
            offenders.append((key, call, missing))
    assert not offenders, f"steering suggestions missing required params: {offenders}"


def test_sessions_are_isolated_per_client() -> None:
    """Two clients must not share an active edata_id (found by dogfooding)."""
    reset_sessions()
    a = get_session(type("Ctx", (), {"client_id": "client-a"})())
    b = get_session(type("Ctx", (), {"client_id": "client-b"})())
    a.set_latest_edata_id("handle-a")
    assert b.get_latest_edata_id() is None
    assert a.get_latest_edata_id() == "handle-a"


def test_session_key_survives_context_without_request() -> None:
    """Probing a Context outside a request must not raise (origin #10)."""

    class _Hostile:
        @property
        def client_id(self):
            raise RuntimeError("outside request context")

        @property
        def session_id(self):
            raise RuntimeError("outside request context")

        @property
        def request_id(self):
            raise RuntimeError("outside request context")

    assert get_session(_Hostile()) is not None


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (TypeError("bad"), "INVALID_INPUT"),
        (ValueError("bad"), "INVALID_VALUE"),
        (FileNotFoundError("bad"), "FILE_NOT_FOUND"),
        (ImportError("bad"), "DEPENDENCY_MISSING"),
        (RuntimeError("bad"), "EXECUTION_ERROR"),
    ],
)
def test_exceptions_map_to_specific_error_codes(exc: Exception, expected: str) -> None:
    """Every failure carries a specific error_code and an agent_action (origin #6)."""
    result = classify_exception_error("run_plot", exc, namespace="plot", function="umap")
    assert result.structured_content["error_code"] == expected
    assert result.structured_content["agent_action"]


def test_kaplan_meier_plot_binds_the_fitted_model() -> None:
    """run_plot('kaplan_meier') must bind the fitter from run_analysis, not the EHRData (origin #5)."""

    async def _test():
        async with Client(mcp) as client:
            await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
            fit = await client.call_tool(
                "run_analysis",
                {
                    "function": "kaplan_meier",
                    "params": {"duration_col": "mort_day_censored", "event_col": "censor_flg"},
                },
            )
            assert fit.structured_content["status"] == "ok"
            plot = await client.call_tool("run_plot", {"function": "kaplan_meier"})
            assert plot.structured_content["status"] == "ok", plot.structured_content
            assert any(type(c).__name__ == "ImageContent" for c in plot.content)

    _run(_test())


def test_plot_without_a_fitted_model_explains_what_to_run() -> None:
    """The unfitted case must steer the agent, not surface an AttributeError."""

    async def _test():
        async with Client(mcp) as client:
            await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
            res = await client.call_tool("run_plot", {"function": "love_plot"})
            struct = res.structured_content
            assert struct["status"] == "error"
            assert "covariate_balance" in struct["reason"]

    _run(_test())


def test_structured_content_is_always_json_serializable() -> None:
    """Analysis results carrying ndarrays must not break the MCP boundary.

    ep.tl.iptw returns a CausalEstimate whose nested params hold ndarrays; before the
    sanitizer these reached structured_content and aborted the whole call.
    """

    async def _test():
        async with Client(mcp) as client:
            await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
            await client.call_tool("run_preprocessing", {"function": "encode", "params": {"autodetect": True}})
            res = await client.call_tool(
                "run_analysis",
                {
                    "function": "iptw",
                    "params": {
                        "treatment": "aline_flg",
                        "outcome": "hosp_exp_flg",
                        "covariates": ["age", "gender_num"],
                    },
                },
            )
            assert res.structured_content["status"] == "ok", res.structured_content
            json.dumps(res.structured_content)

    _run(_test())


def test_fitter_cache_invalidates_when_the_dataset_changes() -> None:
    """A plot must not bind a model fitted against a since-transformed cohort.

    The fitter is stamped with the handle's cache mtime; a later write-through to the
    same edata_id must make it stale rather than silently rendering a survival curve
    for data that no longer exists.
    """

    async def _test():
        async with Client(mcp) as client:
            await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
            await client.call_tool(
                "run_analysis",
                {
                    "function": "kaplan_meier",
                    "params": {"duration_col": "mort_day_censored", "event_col": "censor_flg"},
                },
            )
            # Fitter is fresh: the plot renders.
            assert (await client.call_tool("run_plot", {"function": "kaplan_meier"})).structured_content[
                "status"
            ] == "ok"

            # Transform the same handle, invalidating the fitter.
            await client.call_tool("run_preprocessing", {"function": "encode", "params": {"autodetect": True}})
            res = await client.call_tool("run_plot", {"function": "kaplan_meier"})
            struct = res.structured_content
            assert struct["status"] == "error", struct
            assert "kaplan_meier" in struct["reason"]

    _run(_test())


def test_propensity_overlap_binds_its_fitted_result() -> None:
    """run_plot('propensity_overlap') binds the positivity_check result (origin #5)."""

    async def _test():
        async with Client(mcp) as client:
            await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
            await client.call_tool("run_preprocessing", {"function": "encode", "params": {"autodetect": True}})
            fit = await client.call_tool(
                "run_analysis",
                {
                    "function": "positivity_check",
                    "params": {"treatment": "aline_flg", "covariates": ["age", "gender_num"]},
                },
            )
            assert fit.structured_content["status"] == "ok", fit.structured_content
            plot = await client.call_tool("run_plot", {"function": "propensity_overlap"})
            assert plot.structured_content["status"] == "ok", plot.structured_content
            assert any(type(c).__name__ == "ImageContent" for c in plot.content)

    _run(_test())


def test_unrenderable_figure_reports_instead_of_claiming_success() -> None:
    """A figure no backend can export must error with a reason, not report a bare ok.

    ep.pl.love_plot builds a holoviews Overlay that the matplotlib backend cannot render
    (categorical axis) and that bokeh can only export with selenium installed.
    """

    async def _test():
        async with Client(mcp) as client:
            await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
            await client.call_tool("run_preprocessing", {"function": "encode", "params": {"autodetect": True}})
            await client.call_tool(
                "run_analysis",
                {
                    "function": "covariate_balance",
                    "params": {"treatment": "aline_flg", "covariates": ["age", "gender_num"]},
                },
            )
            res = await client.call_tool("run_plot", {"function": "love_plot"})
            struct = res.structured_content
            if struct["status"] == "ok":
                # A selenium-equipped environment can render it; then it must be a real image.
                assert any(type(c).__name__ == "ImageContent" for c in res.content)
            else:
                assert "selenium" in struct["reason"] or "backend" in struct["reason"]
                assert not any(type(c).__name__ == "ImageContent" for c in res.content)

    _run(_test())
