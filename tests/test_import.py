import subprocess
import sys


def test_import_has_no_global_side_effects():
    code = """
import warnings

import holoviews as hv

import ehrapy

assert not hv.Store.renderers
assert not any(issubclass(category, SyntaxWarning) for _, _, category, _, _ in warnings.filters)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
