# Configuration file for Sphinx documentation builder.

import os
import sys

# Add parent directory to path to import gigacode
sys.path.insert(0, os.path.abspath('..'))


def _read_version() -> str:
    """Read the single-source-of-truth version (pyproject -> VERSION)."""
    version_file = os.path.join(os.path.abspath('..'), 'VERSION')
    if os.path.exists(version_file):
        with open(version_file, encoding='utf-8') as handle:
            return handle.read().strip()
    try:
        from gigacode import __version__  # noqa: WPS433

        return __version__
    except Exception:  # pragma: no cover - docs must never fail on version read
        return '0.0.0'


# -- Project information
project = 'GigaCode'
copyright = '2025, GigaCode Contributors'
author = 'GigaCode Contributors'
release = _read_version()
version = release

# -- General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', '_build_check', 'Thumbs.db', '.DS_Store']

# Optional / heavy imports are mocked so the docs build without the full
# runtime dependency set installed.
autodoc_mock_imports = [
    'torch',
    'sentence_transformers',
    'transformers',
    'sklearn',
    'faiss',
    'fastapi',
    'uvicorn',
    'pydantic',
    'watchdog',
    'prometheus_client',
    'mcp',
    'starlette',
]

# Docstring RST style issues in legacy docstrings should not fail the build;
# all structural warnings (references, toctrees, duplicates) remain errors.
suppress_warnings = [
    'docutils',
]

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
