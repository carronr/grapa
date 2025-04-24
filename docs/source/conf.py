# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import sys
from pathlib import Path
sys.path.insert(0, str(Path('..', "..", "grapa").resolve()))
sys.path.insert(0, str(Path('..', "..").resolve()))

from grapa import __version__ as grapaversion


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

print("grapaversion", type(grapaversion), grapaversion)

project = 'grapa'
copyright = '2025, Romain Carron'
author = 'Romain Carron'
release = grapaversion

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
]
autosummary_generate = True
# autosummary_imported_members = False
autosummary_filename_map = {"grapa.GUI": "grapa.guiapp"}  # duplicate in naming

autodoc_default_flags = ['members']

templates_path = ['_templates']
exclude_patterns = ["**grapa.tests**", "tests"]

# toc_object_entries_show_parents = "domain"  # that's good
modindex_common_prefix = ["grapa."]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinxdoc'
html_theme = 'sphinx_rtd_theme'
# html_theme = 'alabaster'
# html_theme = 'classic'
html_static_path = ['_static']
html_logo = '_static/datareading.png'

# HOW TO USE: in command line, execute:
# sphinx-build -M html docs/source/ docs/build
# or go to folder above, "make html source build"