# Configuration file for Sphinx documentation builder.

import os
import sys

# Add parent directory to path to import gigacode
sys.path.insert(0, os.path.abspath('..'))

# -- Project information
project = 'GigaCode'
copyright = '2025, GigaCode Contributors'
author = 'GigaCode Contributors'
release = '1.0.0'

# -- General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- autodoc configuration
autodoc_member_order = 'bysource'
autoclass_content = 'both'

# -- Napoleon configuration (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}
