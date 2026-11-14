"""Standalone Sphinx configuration for the SpectroChemPy NMR documentation."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


DOCS = Path(__file__).parent
PROJECT = DOCS.parent
sys.path.insert(0, str(PROJECT / "src"))

project = "SpectroChemPy NMR"
copyright = "2014-2026, Laboratoire Catalyse et Spectrochimie (LCS)"

import spectrochempy  # noqa: E402
import spectrochempy_nmr  # noqa: E402

_rtd_tag = os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "")
_release_tag = f"spectrochempy-nmr-v{spectrochempy_nmr.NMRPlugin.version}"
_tagged_release = _rtd_tag == _release_tag
_release_build = bool(os.environ.get("NMR_DOCS_RELEASE")) or _tagged_release
github_ref = os.environ.get("NMR_DOCS_GITHUB_REF", "").strip() or (
    _rtd_tag if _tagged_release else "master"
)
version = spectrochempy_nmr.NMRPlugin.version if _release_build else "Unreleased"
release = version
_header_version = (
    "dev"
    if os.environ.get("READTHEDOCS_VERSION") == "latest" or not _release_build
    else version
)
language = "en"
master_doc = "index"
source_suffix = ".rst"
exclude_patterns = [
    "_build",
    "_static",
    "conf.py",
    "fetch_gallery_data.py",
    "**.ipynb_checkpoints",
    # Sphinx-Gallery renders the paired RST for these downloads.
    "examples/gallery/**/*.ipynb",
    "examples/gallery/**/*.py",
]
templates_path = ["_templates"]

core_docs_url = os.environ.get(
    "NMR_CORE_DOCS_URL",
    (
        "https://www.spectrochempy.fr/spectrochempy-v1.2.0/"
        if _tagged_release
        else "https://www.spectrochempy.fr/latest/"
    ),
).rstrip("/")
if _release_build:
    core_docs_version = core_docs_url.rsplit("/", maxsplit=1)[-1]
    immutable_ref = re.fullmatch(r"[0-9a-f]{40}", github_ref) or re.fullmatch(
        r"spectrochempy-nmr-v\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", github_ref
    )
    if not immutable_ref:
        raise RuntimeError(
            "release builds require a commit SHA or spectrochempy-nmr-vX.Y.Z tag"
        )
    if not re.fullmatch(
        r"(?:spectrochempy-v)?\d+\.\d+(?:\.\d+)?(?:[-+][0-9A-Za-z.-]+)?",
        core_docs_version,
    ):
        raise RuntimeError("release builds require a fixed NMR_CORE_DOCS_URL")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_codeautolink",
    "sphinx_gallery.gen_gallery",
    "nbsphinx",
    "sphinxcontrib.bibtex",
]

codeautolink_concat_default = True
# The public reference documents Experiment under its re-exported name.
codeautolink_inventory_map = {
    f"spectrochempy_nmr.experiment.Experiment.{method}": f"spectrochempy_nmr.Experiment.{method}"
    for method in vars(spectrochempy_nmr.Experiment)
    if not method.startswith("_")
}
codeautolink_inventory_map.update(
    {
        "spectrochempy_nmr._resolve_read.<locals>.read": "spectrochempy_nmr.read",
    }
)

bibtex_bibfiles = ["bibliography.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# The core inventory is deliberately external: this project does not import the
# core documentation tree or use its build scripts.
intersphinx_mapping = {
    "core": (f"{core_docs_url}/", None),
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

nbsphinx_execute = "always"
nbsphinx_custom_formats = {".py": ["jupytext.reads", {"fmt": "py:percent"}]}

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {release}"
html_logo = "_static/scpy.png"
html_favicon = "_static/scpy.ico"
html_static_path = ["_static"]
html_css_files = ["css/nmr.css"]
html_sidebars = {"index": ["sidebar-home.html"]}
html_theme_options = {
    "navbar_align": "left",
    "logo": {"text": f"NMR {_header_version}"},
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button-field"],
    "header_links_before_dropdown": 6,
    "navigation_depth": 4,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "collapse_navigation": False,
    "use_edit_page_button": True,
    "search_bar_text": "Search the NMR documentation...",
    "icon_links": [
        {
            "name": "SpectroChemPy",
            "url": "https://www.spectrochempy.fr/",
            "icon": "fa-solid fa-house",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/spectrochempy/spectrochempy_nmr",
            "icon": "fa-brands fa-github",
        },
    ],
}
html_context = {
    "core_docs_url": core_docs_url,
    "default_mode": "light",
    "display_github": True,
    "github_user": "spectrochempy",
    "github_repo": "spectrochempy_nmr",
    # Set NMR_DOCS_GITHUB_REF to a release tag for historical builds. Release
    # builds reject the moving default unless this is set explicitly.
    "github_version": github_ref,
    "doc_path": "docs",
}

# The public gallery examples are sourced from this checkout. They are rendered
# without execution by default; ``-D plot_gallery=True`` opts into execution.
_examples = PROJECT / "examples"
sphinx_gallery_conf = {
    "examples_dirs": [
        str(_examples / "core" / "c_importer"),
        str(_examples / "processing" / "apodization"),
        str(_examples / "processing" / "nmr"),
        str(_examples / "analysis" / "nmr"),
        str(_examples / "simulation"),
    ],
    "gallery_dirs": [
        "examples/gallery/importer",
        "examples/gallery/apodization",
        "examples/gallery/nmr",
        "examples/gallery/analysis",
        "examples/gallery/simulation",
    ],
    "filename_pattern": r"plot_",
    "plot_gallery": os.environ.get("NMR_DOCS_EXECUTE") == "1"
    or os.environ.get("READTHEDOCS", "").lower() == "true",
    "abort_on_example_error": True,
    "only_warn_on_example_error": False,
    "download_all_examples": False,
    "remove_config_comments": True,
}

if os.environ.get("NMR_DOCS_EXECUTE") == "1":
    sphinx_gallery_conf["plot_gallery"] = True


def _set_section_title(app, pagename, templatename, context, doctree):
    """Label the sidebar and menu with the page's top-level toctree title."""
    env = app.env

    def contains(docname, seen):
        if docname == pagename:
            return True
        if docname in seen:
            return False
        seen.add(docname)
        return any(
            contains(child, seen) for child in env.toctree_includes.get(docname, ())
        )

    for section in env.toctree_includes.get(env.config.root_doc, ()):
        if contains(section, set()):
            context["section_title"] = env.titles[section].astext()
            break


def setup(app):
    from sphinx_gallery.docs_resolv import embed_code_links

    # Let codeautolink handle both galleries and notebooks. The gallery's
    # HTML substitution can otherwise wrap existing links in a second anchor.
    for listener in tuple(app.events.listeners["build-finished"]):
        if listener.handler is embed_code_links:
            app.disconnect(listener.id)

    app.connect("html-page-context", _set_section_title)
