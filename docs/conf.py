#!/usr/bin/env python
# mypy: ignore-errors

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent), str(HERE / "extensions")]

needs_sphinx = "4.3"

# General information about the project.
project = "ehrapy"
copyright = "2021-2024, Lukas Heumos, Theislab"
author = "Lukas Heumos"
github_repo = "ehrapy"

version = "0.7.0"
release = "0.7.0"

extensions = [
    "myst_parser",
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
]

# remove_from_toctrees = ["tutorials/notebooks/*", "api/reference/*"]

# for sharing urls with nice info
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
]
nbsphinx_execute = "never"

templates_path = ["_templates"]
# source_suffix = ".md"

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = True  # for pytorch lightning
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False
numpydoc_show_class_members = False
annotate_defaults = True  # scanpydoc option, look into why we need this
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]

# The master toctree document.
master_doc = "index"

intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "pytorch_lightning": ("https://pytorch-lightning.readthedocs.io/en/stable/", None),
    "pyro": ("http://docs.pyro.ai/en/stable/", None),
    "pymde": ("https://pymde.org/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "lamin": ("https://lamin.ai/docs", None),
}

language = "en"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"
pygments_dark_style = "native"


# -- Options for HTML output -------------------------------------------

# html_show_sourcelink = True
html_theme = "furo"

# Set link name generated in the top bar.
html_title = "ehrapy"
html_logo = "_static/ehrapy_logos/ehrapy_pure.png"

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#003262",
        "color-brand-content": "#003262",
        "admonition-font-size": "var(--font-size-normal)",
        "admonition-title-font-size": "var(--font-size-normal)",
        "code-font-size": "var(--font-size--small)",
    },
    "top_of_page_button": None,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/override.css", "css/sphinx_gallery.css"]
html_show_sphinx = False


nbsphinx_prolog = r"""
.. raw:: html

{{% set docname = env.doc2path(env.docname, base=None).split("/")[-1] %}}

.. raw:: html

    <style>
        p {{
            margin-bottom: 0.5rem;
        }}
        /* Main index page overview cards */
        /* https://github.com/spatialaudio/nbsphinx/pull/635/files */
        .jp-RenderedHTMLCommon table,
        div.rendered_html table {{
        border: none;
        border-collapse: collapse;
        border-spacing: 0;
        font-size: 12px;
        table-layout: fixed;
        color: inherit;
        }}

        body:not([data-theme=light]) .jp-RenderedHTMLCommon tbody tr:nth-child(odd),
        body:not([data-theme=light]) div.rendered_html tbody tr:nth-child(odd) {{
        background: rgba(255, 255, 255, .1);
        }}
    </style>

.. raw:: html

    <div class="admonition note">
        <p class="admonition-title">Note</p>
        <p>
        This page was generated from
        <a class="reference external" href="https://github.com/theislab/ehrapy/tree/{version}/">{docname}</a>.
        Some tutorial content may look better in light mode.
        </p>
    </div>
""".format(version=version, docname="{{ docname|e }}")
nbsphinx_thumbnails = {
    "tutorials/notebooks/ehrapy_introduction": "_static/ehrapy_logos/ehrapy_pure.png",
    "tutorials/notebooks/mimic_2_introduction": "_static/tutorials/catheter.png",
    "tutorials/notebooks/mimic_2_fate": "_static/tutorials/fate.png",
    "tutorials/notebooks/mimic_2_survival_analysis": "_static/tutorials/survival.png",
    "tutorials/notebooks/mimic_2_causal_inference": "_static/tutorials/causal_inference.png",
    "tutorials/notebooks/medcat": "_static/tutorials/nlp.png",
    "tutorials/notebooks/ml_usecases": "_static/tutorials/machine_learning.png",
    "tutorials/notebooks/ontology_mapping": "_static/tutorials/ontology.png",
    "tutorials/notebooks/fhir": "_static/tutorials/fhir.png",
}
