"""Validation and error response helpers for CodeEmbeddingTool.

Provides standardized validation and error formatting for API responses.
"""

from typing import Any, Dict, Optional

from gigacode.response_types import ErrorResponse


def make_error_response(
    message: str,
    buffer_id: Optional[str] = None,
    operation: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a structured error response with context.
    
    Args:
        message: Human-readable error message.
        buffer_id: Buffer ID if applicable.
        operation: Operation name (e.g., 'semantic_search', 'write_code').
        context: Additional context dict.
    
    Returns:
        Dict with ErrorResponse.to_dict() format (includes context field).
    """
    ctx = context or {}
    if buffer_id is not None:
        ctx["buffer_id"] = buffer_id
    if operation:
        ctx["operation"] = operation
    response = ErrorResponse(message=message, context=ctx)
    return response.to_dict()


def validate_search_params(
    query: str,
    top_k: Optional[int] = None,
    max_results: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Validate search parameters and return error dict if invalid.
    
    Args:
        query: Search query string
        top_k: Maximum results to return (optional, 1-10000)
        max_results: Maximum results to scan (optional, 1-100000)
    
    Returns:
        None if valid, error dict if invalid
    """
    if not query or not query.strip():
        return {"status": "error", "message": "query must be a non-empty string."}
    if top_k is not None:
        if not isinstance(top_k, int) or top_k < 1 or top_k > 10_000:
            return {"status": "error", "message": "top_k must be an integer between 1 and 10000."}
    if max_results is not None:
        if not isinstance(max_results, int) or max_results < 1 or max_results > 100_000:
            return {"status": "error", "message": "max_results must be an integer between 1 and 100000."}
    return None
