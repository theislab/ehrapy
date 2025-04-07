"""Based on gist.github.com/MantasVaitkunas/7c16de233812adcb7028."""

import os
import warnings
from typing import Any, Optional

from sphinx.application import Sphinx

__licence__ = "BSD (3 clause)"


def get_github_repo(app: Sphinx, path: str) -> str:  # noqa: D103
    if path.endswith(".ipynb"):
        return str(app.config.github_nb_repo)
    if "auto_examples" in path:
        return str(app.config.github_nb_repo)
    if "auto_tutorials" in path:
        return str(app.config.github_nb_repo)
    return str(app.config.github_repo)


def _html_page_context(
    app: Sphinx, _pagename: str, templatename: str, context: dict[str, Any], doctree: Any | None
) -> None:
    # doctree is None - otherwise viewcode fails
    if templatename != "page.html" or doctree is None:
        return

    if not app.config.github_repo:
        return

    if not app.config.github_nb_repo:
        nb_repo = f"{app.config.github_repo}_notebooks"
        app.config.github_nb_repo = nb_repo

    path = os.path.relpath(doctree.get("source"), app.builder.srcdir)
    repo = get_github_repo(app, path)

    # For sphinx_rtd_theme.
    context["display_github"] = True
    context["github_user"] = "theislab"
    context["github_version"] = "master"
    context["github_repo"] = repo
    context["conf_py_path"] = "/docs/source/"


def setup(app: Sphinx) -> None:  # noqa: D103
    app.add_config_value("github_nb_repo", "", True)
    app.add_config_value("github_repo", "", True)
    app.connect("html-page-context", _html_page_context)
