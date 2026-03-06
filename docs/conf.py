"""Sphinx configuration for nestkit documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nestkit

# -- Project information -------------------------------------------------------

project = "nestkit"
author = "Ettore Rocchi"
copyright = "2026, Ettore Rocchi"
version = nestkit.__version__
release = nestkit.__version__

# -- General configuration -----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_parser",
    "numpydoc",
    "nbsphinx",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Napoleon / numpydoc settings ----------------------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

numpydoc_show_class_members = False

# -- Autodoc settings ----------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

autosummary_generate = True

# -- Intersphinx ---------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output ---------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/nestkit_logo.png"

html_theme_options = {
    "show_prev_next": True,
    "navigation_with_keys": True,
    "search_bar_text": "Search the docs...",
    "logo": {
        "text": "nestkit",
    },
    "announcement": (
        "nestkit v0.1 is now available | "
        "<code>pip install nestkit</code>"
        '<button aria-label="Close" onclick="this.closest(\'.bd-header-announcement\').remove()" '
        'class="announcement-close">&times;</button>'
    ),
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ettorerocchi/nestkit",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/nestkit/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

html_css_files = ["custom.css"]

# -- nbsphinx -----------------------------------------------------------------

nbsphinx_execute = "never"
