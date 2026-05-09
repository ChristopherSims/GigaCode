"""
Integration of Phases 4-10 into CodeEmbeddingTool.

DEPRECATED: This module is a staging file. All phase setup logic has been
moved into CodeEmbeddingTool.__init__ directly, and all public methods live
in gigacode_tool.py as thin delegations.  Keep this file for backward
compatibility only; new code should not depend on these mixins.

The only remaining purpose of this module is to provide the
PhasesIntegrationMixin class (now empty) so existing code that does
``class CodeEmbeddingTool(PhasesIntegrationMixin)`` continues to work.
"""

from typing import Any


class PhasesIntegrationMixin:
    """Deprecated mixin placeholder — all phase logic now lives in CodeEmbeddingTool."""

    def setup_phases_4_10(self) -> None:
        """No-op: phases are initialised directly in CodeEmbeddingTool.__init__."""
        pass


def create_enhanced_tool_class() -> type:
    """Deprecated factory — returns plain CodeEmbeddingTool."""
    from gigacode.gigacode_tool import CodeEmbeddingTool

    return CodeEmbeddingTool
