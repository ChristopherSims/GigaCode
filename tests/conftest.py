"""Pytest configuration for GigaCode tests."""

import sys
from pathlib import Path

# Add the parent directory to the Python path so that src can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))
