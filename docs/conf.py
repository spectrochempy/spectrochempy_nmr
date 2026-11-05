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
import spectrochempy_hypercomplex  # noqa: E402
import spectrochempy_nmr  # noqa: E402

version = spectrochempy_nmr.NMRPlugin.version
release = version
language = "en"
master_doc = "index"
source_suffix = ".rst"
exclude_patterns = ["_build", "_static", "**.ipynb_checkpoints"]
templates_path = ["_templates"]

core_docs_url = os.environ.get(
    "NMR_CORE_DOCS_URL", "https://www.spectrochempy.fr/latest/"
).rstrip("/")
_release_build = os.environ.get("NMR_DOCS_RELEASE")
if _release_build:
    github_ref = os.environ.get("NMR_DOCS_GITHUB_REF", "").strip()
    core_docs_version = core_docs_url.rsplit("/", maxsplit=1)[-1]
    immutable_ref = re.fullmatch(r"[0-9a-f]{40}", github_ref) or re.fullmatch(
        r"spectrochempy-nmr-v\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", github_ref
    )
    if not immutable_ref:
        raise RuntimeError(
            "release builds require a commit SHA or spectrochempy-nmr-vX.Y.Z tag"
        )
    if not os.environ.get("NMR_CORE_DOCS_URL") or core_docs_version == "latest":
        raise RuntimeError("release builds require a fixed NMR_CORE_DOCS_URL")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_gallery.gen_gallery",
]

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

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {release}"
html_logo = "_static/scpy.png"
html_favicon = "_static/scpy.ico"
html_static_path = ["_static"]
html_css_files = ["css/nmr.css"]
html_theme_options = {
    "navbar_align": "left",
    "logo": {"text": "SpectroChemPy NMR"},
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
    "default_mode": "light",
    "display_github": True,
    "github_user": "spectrochempy",
    "github_repo": "spectrochempy_nmr",
    # Set NMR_DOCS_GITHUB_REF to a release tag for historical builds. Release
    # builds reject the moving default unless this is set explicitly.
    "github_version": os.environ.get("NMR_DOCS_GITHUB_REF", "master"),
    "conf_py_path": "/docs/",
}

# The six public examples are sourced from this checkout. They are rendered
# without execution by default; ``-D plot_gallery=True`` opts into execution.
_examples = PROJECT / "examples"
sphinx_gallery_conf = {
    "examples_dirs": [
        str(_examples / "core" / "c_importer"),
        str(_examples / "processing" / "apodization"),
        str(_examples / "processing" / "nmr"),
    ],
    "gallery_dirs": [
        "examples/gallery/importer",
        "examples/gallery/apodization",
        "examples/gallery/nmr",
    ],
    "filename_pattern": r"plot_",
    "ignore_pattern": r"(^|/)(plot_processing_cp_nmr|plot_processing_nmr_2d)\.py$",
    "plot_gallery": os.environ.get("NMR_DOCS_EXECUTE") == "1"
    or os.environ.get("READTHEDOCS", "").lower() == "true",
    "abort_on_example_error": True,
    "only_warn_on_example_error": False,
    "download_all_examples": False,
    "remove_config_comments": True,
    "reference_url": {"spectrochempy": core_docs_url + "/"},
}

if os.environ.get("NMR_DOCS_EXECUTE") == "1":
    sphinx_gallery_conf["plot_gallery"] = True
