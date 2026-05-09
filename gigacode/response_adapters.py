"""Response adapters for SearchService integration.

Adapts SearchService responses to CodeEmbeddingTool format for unified response handling.
"""

from typing import Any, Dict, Optional

from gigacode.response_types import (
    ClusterItem,
    ClusterResponse,
    ResponseStatus,
    SearchMatch,
    SearchResponse,
)

__all__ = [
    "adapt_search_response",
    "adapt_file_response",
    "adapt_cluster_response",
    "adapt_duplicate_response",
]


def adapt_search_response(
    service_result: Any,
    offset: int = 0,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Adapt SearchService response to CodeEmbeddingTool format.

    Args:
        service_result: SearchService.SearchResponse or error dict
        offset: Results offset for slicing
        top_k: Number of results to return (if None, return all)

    Returns:
        Dict in CodeEmbeddingTool response format
    """
    # If it's already an error dict, return as-is
    if isinstance(service_result, dict) and service_result.get("status") == "error":
        return service_result

    # Convert SearchService.SearchResponse to CodeEmbeddingTool format
    if hasattr(service_result, "to_dict"):
        service_dict = service_result.to_dict()
    else:
        service_dict = service_result

    # Extract matches and apply offset/slicing
    matches = service_dict.get("matches", [])
    if offset > 0 or top_k:
        end = offset + top_k if top_k else None
        matches = matches[offset:end]

    # Convert SearchService.SearchMatch to CodeEmbeddingTool.SearchMatch
    converted_matches = []
    for idx, match_dict in enumerate(matches):
        if isinstance(match_dict, dict):
            # Build CodeEmbeddingTool.SearchMatch
            converted = SearchMatch(
                file=match_dict.get("file", ""),
                start_line=match_dict.get("start_line", 0),
                end_line=match_dict.get("end_line", 0),
                score=float(match_dict.get("score", 0.0)),
                doc_id=match_dict.get("doc_id", idx),  # Use index if no doc_id
                type=match_dict.get("type"),
                name=match_dict.get("name"),
                match_type=match_dict.get("match_type", "semantic"),
            )
            converted_matches.append(converted)
        else:
            # Already a SearchMatch object
            converted_matches.append(match_dict)

    # Build response in CodeEmbeddingTool format
    response = SearchResponse(
        status=ResponseStatus.OK,
        matches=converted_matches,
        cached=service_dict.get("cache_hit", False),
    )
    return response.to_dict()


def adapt_file_response(
    service_result: Any,
) -> Dict[str, Any]:
    """Adapt SearchService look_for_file response to CodeEmbeddingTool format.

    SearchService returns: {"status", "files", "count", "pattern"}
    CodeEmbeddingTool expects: {"status", "file_location", "absolute_path", "match_type"}
    """
    if isinstance(service_result, dict):
        if service_result.get("status") == "error":
            return service_result

        files = service_result.get("files", [])
        if not files:
            return {"status": "error", "message": "File not found"}
        elif len(files) == 1:
            # Single match
            return {
                "status": "ok",
                "file_location": files[0],
                "match_type": "found",
            }
        else:
            # Multiple matches
            return {
                "status": "ok",
                "candidates": files,
                "match_type": "multiple",
                "message": f"Found {len(files)} matching files.",
            }
    return service_result


def adapt_cluster_response(
    service_result: Any,
) -> Dict[str, Any]:
    """Adapt SearchService ClusterResult to CodeEmbeddingTool format.

    SearchService returns ClusterResult with dict of clusters
    CodeEmbeddingTool expects list of ClusterItem with file/start_line/end_line/size/avg_score
    """
    if isinstance(service_result, dict) and service_result.get("status") == "error":
        return service_result

    # Convert ClusterResult to dict if needed
    if hasattr(service_result, "to_dict"):
        service_dict = service_result.to_dict()
    else:
        service_dict = service_result

    clusters_dict = service_dict.get("clusters", {})
    cluster_list = []

    # Convert dict-based clusters to list format with aggregated metadata
    for _cluster_id, chunks in clusters_dict.items():
        if chunks:
            first_chunk = chunks[0]
            last_chunk = chunks[-1]
            # Calculate average score from chunk scores
            scores = [
                c.get("score", 0.0) for c in chunks if isinstance(c.get("score"), (int, float))
            ]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            cluster_list.append(
                ClusterItem(
                    file=first_chunk.get("file", ""),
                    start_line=first_chunk.get("start_line", 0),
                    end_line=last_chunk.get("end_line", 0),
                    size=len(chunks),
                    avg_score=avg_score,
                )
            )

    response = ClusterResponse(
        status=ResponseStatus.OK,
        clusters=cluster_list,
    )
    return response.to_dict()


def adapt_duplicate_response(
    service_result: Any,
) -> Dict[str, Any]:
    """Adapt SearchService DuplicateResult to CodeEmbeddingTool format.

    Both formats are similar, just ensure proper dict conversion.
    """
    if isinstance(service_result, dict):
        if service_result.get("status") == "error":
            return service_result
        return service_result

    # Convert DuplicateResult to dict if needed
    if hasattr(service_result, "to_dict"):
        return service_result.to_dict()

    return service_result
