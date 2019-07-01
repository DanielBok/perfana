# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'Perfana'
copyright = '2019, Daniel'
author = 'Daniel'

# The full version, including alpha/beta/rc tags
try:
    import perfana

    revision = perfana.__version__
    version = revision.split('+')[0]
except ModuleNotFoundError:
    version = '0.0.0'
    revision = '0.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',  # this may be loaded after napoleon
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    # This will specify a canonical URL meta link element to tell search engines which URL
    # should be ranked as the primary URL for your documentation. This is important if you
    # have multiple URLs that your documentation is available through. The URL points to
    # the root path of the documentation and requires a trailing slash.
    'canonical_url': '',

    # Only display the logo image, do not display the project name at the top of the sidebar
    'logo_only': False,

    # If True, the version number is shown at the top of the sidebar.
    'display_version': True,

    # Location to display Next and Previous buttons. This can be either bottom, top, both , or None.
    'prev_next_buttons_location': 'bottom',

    # Add an icon next to external links.
    'style_external_links': False,

    # Changes how to view files when using display_github, display_gitlab, etc. When using
    # GitHub or GitLab this can be: blob (default), edit, or raw. On Bitbucket, this can be
    # either: view (default) or edit.
    # 'vcs_pageview_mode': 'edit',

    # Changes the background of the search area in the navigation bar. The value can be anything
    # valid in a CSS background property.
    # 'style_nav_header_background': 'black',

    # TOC options

    # With this enabled, navigation entries are not expandable – the [+] icons next to
    # each entry are removed.
    'collapse_navigation': True,

    # Scroll the navigation with the main page content as you scroll the page.
    'sticky_navigation': True,

    # The maximum depth of the table of contents tree. Set this to -1 to allow unlimited depth.
    'navigation_depth': 4,

    # Specifies if the navigation includes hidden table(s) of contents – that is,
    # any toctree directive that is marked with the :hidden: option.
    'includehidden': True,

    # When enabled, page subheadings are not included in the navigation.
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
