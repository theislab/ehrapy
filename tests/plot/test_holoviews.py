import subprocess
import sys

import nbclient
import nbformat
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


def test_load_hv_extensions_keeps_inline_figures(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MPLBACKEND", raising=False)
    cells = [
        "import holoviews as hv\n"
        "from ehrapy.plot._holoviews import load_hv_extensions\n"
        "load_hv_extensions()(lambda: None)()",
        "import matplotlib.pyplot as plt\nplt.plot([1, 2])",
    ]
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(cell) for cell in cells])
    nbclient.NotebookClient(nb, kernel_name="python3", timeout=600).execute()

    assert any("image/png" in output.get("data", {}) for output in nb.cells[-1].outputs)
