#!/usr/bin/env python
# mypy: ignore-errors

import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent), str(HERE / "extensions")]

needs_sphinx = "8.0"

info = metadata("ehrapy")
project_name = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
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
]

ogp_site_url = "https://ehrapy.readthedocs.io/en/latest/"
ogp_image = "https://ehrapy.readthedocs.io/en/latest//_static/logo.png"

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
    "tutorials/notebooks/diabetic_retinopathy_fate_mapping.ipynb",
]
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

autodoc_mock_imports = ["scipy.linalg.triu"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
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
}
nitpick_ignore = [
    ("py:class", "matplotlib.axes.Axes"),
    ("py:class", "leidenalg.RBConfigurationVertexPartition"),
    ("py:class", "leidenalg.VertexPartition.MutableVertexPartition"),
    ("py:class", "cycler.Cycler"),
    ("py:func", "leidenalg.find_partition"),
    ("py:class", "CAT"),
    ("py:class", "ehrapy.tools.annotate_text.CAT"),
    ("py:class", "tableone.TableOne"),
    ("py:class", "dowhy.causal_estimator.CausalEstimate"),
    ("py:class", "DotPlot"),
    ("py:class", "MatrixPlot"),
    ("py:class", "StackedViolin"),
    ("py:class", "ehrapy.plot.DotPlot"),
    ("py:class", "ehrapy.plot.StackedViolin"),
    ("py:class", "ehrapy.tools._scanpy_tl_api.TypeAliasType"),
    ("py:func", "ehrapy.pl.matrixplot"),
    ("py:func", "ehrapy.pl.tracksplot"),
    ("py:class", "scanpy.plotting._utils._AxesSubplot"),
    ("py:func", "IPython.display.set_matplotlib_formats"),
    ("py:class", "matplotlib.colorbar.ColorbarBase"),
    ("py:class", "scanpy.neighbors._types.KnnTransformerLike"),
    ("py:class", "statsmodels.genmod.generalized_linear_model.GLMResultsWrapper"),
    ("py:class", "dask_ml.preprocessing.MinMaxScaler"),
    ("py:class", "dask_ml.preprocessing.QuantileTransformer"),
    ("py:class", "dask_ml.preprocessing.RobustScaler"),
    ("py:class", "dask_ml.preprocessing.StandardScaler"),
    ("py:class", "pathlib._local.Path"),
    ("py:data", "typing.Union"),
]
autodoc_type_aliases = {"CAT": "Any"}

typehints_defaults = "comma"

pygments_style = "sphinx"
pygments_dark_style = "native"

html_theme = "scanpydoc"
html_title = "ehrapy"
html_logo = "_static/ehrapy_logos/ehrapy_pure.png"
html_theme_options = {}
html_static_path = ["_static"]
html_css_files = ["css/overwrite.css", "css/sphinx_gallery.css"]
html_show_sphinx = False

nbsphinx_thumbnails = {
    "tutorials/notebooks/ehrapy_introduction": "_static/ehrapy_logos/ehrapy_pure.png",
    "tutorials/notebooks/mimic_2_introduction": "_static/tutorials/catheter.png",
    "tutorials/notebooks/mimic_2_fate": "_static/tutorials/fate.png",
    "tutorials/notebooks/mimic_2_survival_analysis": "_static/tutorials/survival.png",
    "tutorials/notebooks/mimic_2_effect_estimation": "_static/tutorials/effect_estimation.png",
    "tutorials/notebooks/mimic_2_causal_inference": "_static/tutorials/causal_inference.png",
    "tutorials/notebooks/ml_usecases": "_static/tutorials/machine_learning.png",
    "tutorials/notebooks/ontology_mapping": "_static/tutorials/ontology.png",
    "tutorials/notebooks/fhir": "_static/tutorials/fhir.png",
    "tutorials/notebooks/cohort_tracking": "_static/tutorials/cohort_tracking.png",
    "tutorials/notebooks/bias": "_static/tutorials/bias.png",
    "tutorials/notebooks/out_of_core": "_static/tutorials/out_of_core.png",
    "tutorials/notebooks/patient_trajectory": "_static/tutorials/patient_trajectory.png",
}
