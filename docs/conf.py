#!/usr/bin/env python
# mypy: ignore-errors

import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

needs_sphinx = "8.0"

info = metadata("ehrapy")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}"
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]
release = info["Version"]
github_repo = "ehrapy"
language = "en"
master_doc = "index"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",  # needs to be after napoleon
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
    "sphinx_remove_toctrees",
    "sphinx_design",
    "sphinx_tabs.tabs",
    "sphinx_issues",
    "sphinxcontrib.bibtex",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    "sphinx_sitemap",
    "sphinx_llms_txt",
    "sphinx_reredirects",
]

html_baseurl = "https://ehrapy.readthedocs.io/en/stable/"
# html_baseurl already contains the language and version
sitemap_url_scheme = "{link}"
sitemap_excludes = ["search.html", "genindex.html", "py-modindex.html"]
ogp_site_url = html_baseurl
ogp_image = f"{html_baseurl}_static/ehrapy_logos/ehrapy_pure.png"
llms_txt_summary = info["Summary"]
llms_txt_exclude = ["changelog", "references", "contributing"]
llms_txt_uri_template = "{base_url}{docname}.html"
llms_txt_full_file = False
# llms_txt_full_file is ignored while _sources exists (sphinx-llms-txt 0.7.1), so cap it away
llms_txt_full_max_size = 0
llms_txt_full_size_policy = "info_skip"

# nbsphinx specific settings
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "auto_*/**.ipynb",
    "auto_*/**.md5",
    "auto_*/**.py",
    "**.ipynb_checkpoints",
    "tutorials/notebooks/README.md",
    "tutorials/notebooks/imputation_nb.ipynb",
]
redirects = {
    **{
        f"tutorials/notebooks/{name}": "../index.html"
        for name in (
            "fhir",
            "out_of_core",
            "patient_trajectory",
            "ontology_mapping",
            "diabetic_retinopathy_fate_mapping",
        )
    },
    **{f"tutorials/notebooks/{name}": "cohort.html" for name in ("cohort_tracking", "imputation_nb")},
    "tutorials/notebooks/mimic_2_introduction": "subgroups.html",
    "tutorials/notebooks/mimic_2_survival_analysis": "survival.html",
    "tutorials/notebooks/mimic_2_fate": "trajectories.html",
    **{
        f"tutorials/notebooks/{name}": "causal.html"
        for name in ("mimic_2_causal_inference", "mimic_2_effect_estimation")
    },
}

nbsphinx_execute = "never"
nb_execution_mode = "off"

templates_path = ["_templates"]
bibtex_bibfiles = ["references.bib"]
nitpicky = True  # Warn about broken links

suppress_warnings = ["toc.not_included", "toc.excluded", "mystnb.unknown_mime_type"]
# source_suffix = ".md"

autosummary_generate = True
autosummary_imported_members = True
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False
numpydoc_show_class_members = False
annotate_defaults = True
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]

autodoc_mock_imports = [
    "scipy.linalg.triu",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pynndescent": ("https://pynndescent.readthedocs.io/en/latest/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "torch": ("https://docs.pytorch.org/docs/main", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "pytorch_lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "pymde": ("https://pymde.org/", None),
    "lamin": ("https://docs.lamin.ai", None),
    "lifelines": ("https://lifelines.readthedocs.io/en/latest/", None),
    "statsmodels": ("https://www.statsmodels.org/stable", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "ehrdata": ("https://ehrdata.readthedocs.io/en/latest/", None),
    "holoviews": ("https://holoviews.org/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    "igraph": ("https://python.igraph.org/en/stable/api/", None),
}
nitpick_ignore = [
    ("py:class", "matplotlib.axes.Axes"),
    ("py:class", "seaborn.matrix.ClusterGrid"),
    ("py:class", "ehrapy._types.Empty"),
    ("py:class", "cycler.Cycler"),
    ("py:class", "tableone.TableOne"),
    ("py:class", "DotPlot"),
    ("py:class", "MatrixPlot"),
    ("py:class", "StackedViolin"),
    ("py:class", "ehrapy.plot.DotPlot"),
    ("py:class", "ehrapy.plot.StackedViolin"),
    ("py:class", "ehrapy.tools._scanpy_tl_api.TypeAliasType"),
    ("py:class", "ehrapy.preprocessing._summarize_measurements.Statistic"),
    ("py:func", "ehrapy.pl.matrixplot"),
    ("py:func", "ehrapy.pl.tracksplot"),
    ("py:class", "scanpy.plotting._utils._AxesSubplot"),
    ("py:func", "IPython.display.set_matplotlib_formats"),
    ("py:func", "matplotlib.cm.get_cmap"),
    ("py:class", "matplotlib.colorbar.ColorbarBase"),
    ("py:class", "scanpy.neighbors._types.KnnTransformerLike"),
    ("py:class", "statsmodels.genmod.generalized_linear_model.GLMResultsWrapper"),
    ("py:class", "pathlib._local.Path"),
    ("py:data", "typing.Union"),
    ("py:class", "pandas.core.frame.DataFrame"),
]

typehints_defaults = "comma"
always_use_bars_union = True

pygments_style = "sphinx"
pygments_dark_style = "native"

html_theme = "scanpydoc"
html_title = "ehrapy"
html_logo = "_static/ehrapy_logos/ehrapy_pure.png"
html_theme_options = {
    "show_toc_level": 2,
}
html_static_path = ["_static"]
html_css_files = ["css/overwrite.css", "css/sphinx_gallery.css"]
html_show_sphinx = False

nbsphinx_thumbnails = {
    "tutorials/notebooks/ehrapy_introduction": "_static/ehrapy_logos/ehrapy_pure.png",
    "tutorials/notebooks/trajectories": "_static/tutorials/trajectories.png",
    "tutorials/notebooks/survival": "_static/tutorials/survival.png",
    "tutorials/notebooks/causal": "_static/tutorials/causal_inference.png",
    "tutorials/notebooks/cohort": "_static/tutorials/cohort_tracking.png",
    "tutorials/notebooks/subgroups": "_static/tutorials/subgroups.png",
    "tutorials/notebooks/bias": "_static/tutorials/bias.png",
    "tutorials/notebooks/longitudinal_with_ehrapy": "_static/tutorials/longitudinal_with_ehrapy.png",
    "tutorials/notebooks/prediction": "_static/tutorials/machine_learning.png",
}


# Redirect broken parameter annotation classes
qualname_overrides = {
    "pandas.core.series.Series": "pandas.Series",
    "pandas.core.frame.DataFrame": "pandas.DataFrame",
}
