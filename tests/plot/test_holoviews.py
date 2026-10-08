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
import matplotlib

from ehrapy.plot._holoviews import load_hv_extensions

matplotlib.use("svg")
{setup}
extension = hv.extension


def notebook_extension(*backends):
    # in notebooks, holoviews switches matplotlib to agg when it loads its matplotlib extension
    extension(*backends)
    matplotlib.pyplot.switch_backend("agg")


hv.extension = notebook_extension


@load_hv_extensions()
def plot():
    return hv.Store.current_backend

assert plot() == {expected_backend!r}
assert {{"bokeh", "matplotlib"}} <= set(hv.Store.renderers)
assert matplotlib.get_backend() == "svg"
"""
    subprocess.run([sys.executable, "-c", code], check=True)
