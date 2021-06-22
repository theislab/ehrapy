#!/usr/bin/env python
# mypy: ignore-errors
# ehrapy documentation build configuration file
import os
import sys
from pathlib import Path

from sphinx.application import Sphinx
from sphinx_gallery.gen_gallery import DEFAULT_GALLERY_CONF

from docs.utils import _get_thumbnails, MaybeMiniGallery

HERE = Path(__file__).parent
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("_ext"))

# General information about the project.
project = "ehrapy"
copyright = "2021, Lukas Heumos"
author = "Lukas Heumos"
github_repo = "ehrapy"

version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------
# Add 'sphinx_automodapi.automodapi' if you want to build modules
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_gallery.load_style",
    "nbsphinx",
    "sphinxcontrib.bibtex",
    "typed_returns",
    "sphinx_click",
    "sphinx_rtd_dark_mode",
]
intersphinx_mapping = dict(  # noqa: C408
    python=("https://docs.python.org/3", None),
    numpy=("https://docs.scipy.org/doc/numpy/", None),
    statsmodels=("https://www.statsmodels.org/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/reference/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    scanpy=("https://scanpy.readthedocs.io/en/stable/", None),
    matplotlib=("https://matplotlib.org/", None),
    seaborn=("https://seaborn.pydata.org/", None),
    joblib=("https://joblib.readthedocs.io/en/latest/", None),
    networkx=("https://networkx.org/documentation/stable/", None),
    dask=("https://docs.dask.org/en/latest/", None),
    numba=("https://numba.readthedocs.io/en/stable/", None),
    xarray=("https://xarray.pydata.org/en/stable/", None),
)

default_dark_mode = True

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = None

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "auto_*/**.ipynb",
    "auto_*/**.md5",
    "auto_*/**.py",
    "**.ipynb_checkpoints",
]
suppress_warnings = ["download.not_readable"]
pygments_style = "sphinx"

html_css_files = ["custom_cookietemple.css"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/img/squidpy_horizontal.png"  # TODO
html_theme_options = {"navigation_depth": 4, "logo_only": True}
html_show_sphinx = False

autosummary_generate = True
autodoc_member_order = "groupwise"
autodoc_typehints = "signature"
autodoc_docstring_signature = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False

# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

# spelling
spelling_lang = "en_US"
spelling_warning = True
spelling_word_list_filename = "spelling_wordlist.txt"
spelling_add_pypi_package_names = True
spelling_show_suggestions = True
spelling_exclude_patterns = ["references.rst"]
# see: https://pyenchant.github.io/pyenchant/api/enchant.tokenize.html
spelling_filters = [
    "enchant.tokenize.URLFilter",
    "enchant.tokenize.EmailFilter",
    "docs.source.utils.ModnameFilter",
    "docs.source.utils.SignatureFilter",
]

nbsphinx_thumbnails = {**_get_thumbnails("auto_tutorials"), **_get_thumbnails("auto_examples")}
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png', 'pdf'}",  # correct figure resize
    "--InlineBackend.rc={'figure.dpi': 96}",
]
nbsphinx_prolog = r"""
{% set docname = 'docs/source/' + env.doc2path(env.docname, base=None) %}
.. raw:: html

    <div class="binder-badge docutils container">
        <a class="reference external image-reference"
           href="https://mybinder.org/v2/gh/theislab/squidpy_notebooks/{{ env.config.release|e }}?filepath={{ docname|e }}">
        <img alt="Launch binder" src="https://mybinder.org/badge_logo.svg" width="150px">
        </a>
    </div>
"""  # noqa: E501

# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "ehrapydoc"

# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ("letterpaper" or "a4paper").
    #
    # "papersize": "letterpaper",
    # The font size ("10pt", "11pt" or "12pt").
    #
    # "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    #
    # "preamble": "",
    # Latex figure (float) alignment
    #
    # "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "ehrapy.tex",
        "ehrapy Documentation",
        "Lukas Heumos",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        master_doc,
        "ehrapy",
        "ehrapy Documentation",
        [author],
        1,
    )
]

autodoc_typehints = "description"

# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "ehrapy",
        "ehrapy Documentation",
        author,
        "ehrapy",
        "One line description of project.",
        "Miscellaneous",
    ),
]


def setup(app: Sphinx) -> None:
    DEFAULT_GALLERY_CONF["src_dir"] = str(HERE)
    DEFAULT_GALLERY_CONF["backreferences_dir"] = "gen_modules/backreferences"
    DEFAULT_GALLERY_CONF["download_all_examples"] = False
    DEFAULT_GALLERY_CONF["show_signature"] = False
    DEFAULT_GALLERY_CONF["log_level"] = {"backreference_missing": "info"}
    DEFAULT_GALLERY_CONF["gallery_dirs"] = ["auto_examples", "auto_tutorials"]
    DEFAULT_GALLERY_CONF["default_thumb_file"] = "docs/source/_static/img/squidpy_vertical.png"

    app.add_config_value("sphinx_gallery_conf", DEFAULT_GALLERY_CONF, "html")
    app.add_directive("minigallery", MaybeMiniGallery)
    app.add_css_file("css/custom.css")
    app.add_css_file("css/sphinx_gallery.css")
    app.add_css_file("css/nbsphinx.css")
    app.add_css_file("css/dataframe.css")  # had to add this manually
