import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("setup", "expected_backend"),
    [("", "bokeh"), ("hv.extension('matplotlib')", "matplotlib")],
)
def test_load_hv_extensions(setup: str, expected_backend: str):
    code = f"""
import holoviews as hv

from ehrapy.plot._holoviews import load_hv_extensions

{setup}

@load_hv_extensions()
def plot():
    return hv.Store.current_backend

assert plot() == {expected_backend!r}
assert {{"bokeh", "matplotlib"}} <= set(hv.Store.renderers)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
