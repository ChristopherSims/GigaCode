"""
Phase 6: Diff-Aware Search

Smart search that focuses on modified files and their dependencies,
avoiding irrelevant results from unrelated parts of codebase.

Key features:
- Scope-based search (changes | changes+deps | all)
- Dependency graph integration
- Performance metrics and speedup calculation
- Weighted result ranking by edit context
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum

from gigacode.metrics import timer, log_metric
from gigacode.dependency_graph import DependencyGraph


class SearchScope(Enum):
    """Scope for diff-aware search."""
    CHANGES = "changes"  # Only edited files
    CHANGES_AND_DEPS = "changes+deps"  # Edited files + their dependencies
    ALL = "all"  # Entire codebase


@dataclass
class DiffAwareSearchResult:
    """Single search result with context."""
    file: str
    line: int
    score: float
    snippet: str
    context: Dict[str, any] = field(default_factory=dict)  # Edit state, dependencies, etc.


@dataclass
class DiffAwareSearchResponse:
    """Response from diff-aware search."""
    query: str
    scope_used: SearchScope
    files_searched: List[str]
    files_skipped: List[str]
    results: List[DiffAwareSearchResult]
    performance: Dict[str, any] = field(default_factory=dict)


class DiffAwareSearch:
    """
    Executes search operations scoped to modified files and dependencies.
    """
    
    def __init__(
        self,
        search_service,
        dependency_graph: DependencyGraph,
        buffer_manager,
    ):
        """
        Initialize diff-aware search.
        
        Args:
            search_service: SearchService instance for semantic/lexical search
            dependency_graph: DependencyGraph instance
            buffer_manager: BufferManager instance for buffer state
        """
        self.search_service = search_service
        self.dependency_graph = dependency_graph
        self.buffer_manager = buffer_manager
    
    def search_since_last_edit(
        self,
        buffer_id: str,
        query: str,
        scope: SearchScope = SearchScope.CHANGES_AND_DEPS,
        include_deleted: bool = False,
        top_k: int = 10,
    ) -> DiffAwareSearchResponse:
        """
        Search only modified files and their dependencies.
        
        Args:
            buffer_id: Target buffer
            query: Search query
            scope: Search scope (changes | changes+deps | all)
            include_deleted: Include files deleted in dirty queue?
            top_k: Max results to return
            
        Returns:
            DiffAwareSearchResponse with scoped results
        """
        with timer("search_since_last_edit"):
            # Get buffer and its dirty state
            # Try multiple possible API names for compatibility
            buffer = None
            if hasattr(self.buffer_manager, '_get_buffer_info'):
                buffer = self.buffer_manager._get_buffer_info(buffer_id)
            elif hasattr(self.buffer_manager, 'get_buffer_info'):
                buffer = self.buffer_manager.get_buffer_info(buffer_id)
            elif hasattr(self.buffer_manager, 'get'):
                buffer = self.buffer_manager.get(buffer_id)
            dirty_files = self._get_dirty_files(buffer, include_deleted)
            
            # Determine files to search
            if scope == SearchScope.ALL:
                files_to_search = None  # Search all
                files_skipped = []
            elif scope == SearchScope.CHANGES:
                files_to_search = dirty_files
                files_skipped = self._get_all_files(buffer)
            else:  # CHANGES_AND_DEPS
                files_to_search = self._expand_with_dependencies(
                    buffer_id, dirty_files
                )
                files_skipped = self._get_all_files(buffer)
            
            # Execute search with file filter
            all_results = self._search_with_scope(
                buffer_id,
                query,
                files_to_search,
            )
            
            # Enhance results with context information
            results = self._enhance_results(
                all_results,
                buffer,
                dirty_files,
                top_k,
            )
            
            # Compute performance metrics
            perf = self._compute_performance(
                query,
                files_to_search,
                dirty_files,
                buffer,
            )
            
            response = DiffAwareSearchResponse(
                query=query,
                scope_used=scope,
                files_searched=files_to_search or self._get_all_files(buffer),
                files_skipped=files_skipped,
                results=results,
                performance=perf,
            )
            
            log_metric("search_since_last_edit", {
                "scope": scope.value,
                "files_searched": len(response.files_searched),
                "files_skipped": len(files_skipped),
                "results": len(results),
                "speedup": perf.get("speedup_factor", 1.0),
            })
            
            return response
    
    def _get_dirty_files(self, buffer: dict, include_deleted: bool = False) -> Set[str]:
        """Get list of files in dirty queue (edited/added/deleted)."""
        dirty_files = set()
        
        if 'dirty_queue' in buffer:
            for entry in buffer['dirty_queue']:
                if include_deleted or entry.get('operation') != 'delete':
                    dirty_files.add(entry.get('file'))
        
        # Also include files with active edits
        if 'edits' in buffer:
            for edit in buffer['edits']:
                dirty_files.add(edit.get('file'))
        
        return dirty_files
    
    def _expand_with_dependencies(
        self,
        buffer_id: str,
        dirty_files: Set[str],
    ) -> Set[str]:
        """
        Expand dirty files with their direct dependencies.
        
        Args:
            buffer_id: Target buffer
            dirty_files: Currently edited files
            
        Returns:
            Set of dirty files + their dependencies
        """
        expanded = set(dirty_files)
        
        for file in dirty_files:
            # Get files that import this file (incoming/dependents)
            if hasattr(self.dependency_graph, 'get_dependents'):
                dependents = self.dependency_graph.get_dependents(buffer_id, file)
            elif hasattr(self.dependency_graph, 'get_dependencies'):
                dependents = self.dependency_graph.get_dependencies(file, direction="incoming")
            else:
                dependents = []
            expanded.update(dependents)

            # Get files that this file imports (outgoing)
            if hasattr(self.dependency_graph, 'get_dependencies'):
                dependencies = self.dependency_graph.get_dependencies(file, direction="outgoing")
            else:
                dependencies = []
            expanded.update(dependencies)

        return expanded
    
    def _search_with_scope(
        self,
        buffer_id: str,
        query: str,
        files_to_search: Optional[List[str]],
    ) -> List[Dict]:
        """Execute search with optional file filtering."""
        # Call underlying search service - adapt to actual API
        if hasattr(self.search_service, 'semantic_search'):
            result = self.search_service.semantic_search(
                buffer_id,
                query,
                top_k=50,  # Get more results; we'll filter
            )
            # Handle SearchResponse dataclass or dict
            if hasattr(result, 'to_dict'):
                result = result.to_dict()
            elif hasattr(result, 'matches'):
                result = {"matches": result.matches}
            results = result.get("matches", []) if isinstance(result, dict) else []
        elif hasattr(self.search_service, 'search'):
            results = self.search_service.search(
                buffer_id,
                query,
                top_k=50,
            )
        else:
            results = []

        # Filter results by file if scope is limited
        if files_to_search is not None:
            files_set = set(files_to_search)
            results = [r for r in results if r.get('file') in files_set]

        return results
    
    def _enhance_results(
        self,
        results: List[Dict],
        buffer: dict,
        dirty_files: Set[str],
        top_k: int,
    ) -> List[DiffAwareSearchResult]:
        """
        Enhance search results with context information.
        
        Args:
            results: Raw search results
            buffer: Buffer state
            dirty_files: Files currently being edited
            top_k: Max results to return
            
        Returns:
            Enhanced results sorted by relevance + context
        """
        enhanced = []
        
        for result in results[:top_k]:
            file = result.get('file')
            
            # Build context
            context = {
                'is_edited': file in dirty_files,
                'edit_step': self._get_edit_step(buffer, file),
            }
            
            # Add dependency information
            if hasattr(self.dependency_graph, 'get_dependencies'):
                deps = self.dependency_graph.get_dependencies(
                    file,
                    direction="outgoing",
                )
            else:
                deps = []

            context['is_dependency_of'] = []
            for f in dirty_files:
                if hasattr(self.dependency_graph, 'get_dependents'):
                    if f in self.dependency_graph.get_dependents(buffer.get('buffer_id'), f):
                        context['is_dependency_of'].append(f)
                elif hasattr(self.dependency_graph, 'get_dependencies'):
                    incoming = self.dependency_graph.get_dependencies(f, direction="incoming")
                    if file in incoming:
                        context['is_dependency_of'].append(f)
            
            enhanced_result = DiffAwareSearchResult(
                file=file,
                line=result.get('line', 0),
                score=result.get('score', 0.0),
                snippet=result.get('snippet', ''),
                context=context,
            )
            enhanced.append(enhanced_result)
        
        # Sort by: is_edited (yes=higher), then score
        enhanced.sort(
            key=lambda r: (r.context.get('is_edited', False), r.score),
            reverse=True,
        )
        
        return enhanced
    
    def _get_edit_step(self, buffer: dict, file: str) -> Optional[int]:
        """Get which edit step this file was modified in."""
        if 'edits' in buffer:
            for i, edit in enumerate(buffer['edits']):
                if edit.get('file') == file:
                    return i
        return None
    
    def _get_all_files(self, buffer: dict) -> List[str]:
        """Get all files in buffer (for skipped list)."""
        if 'files' in buffer:
            return buffer['files']
        return []
    
    def _compute_performance(
        self,
        query: str,
        files_to_search: Optional[List[str]],
        dirty_files: Set[str],
        buffer: dict,
    ) -> Dict[str, any]:
        """
        Compute performance metrics for this search.
        
        Args:
            query: Search query
            files_to_search: Files in scope (None = all)
            dirty_files: Edited files
            buffer: Buffer state
            
        Returns:
            Performance metrics dict
        """
        all_files = self._get_all_files(buffer)
        
        if files_to_search is None:
            chunks_searched = len(all_files) * 10  # Rough estimate
            speedup = 1.0
        else:
            chunks_searched = len(files_to_search) * 10
            total_chunks = len(all_files) * 10
            speedup = max(1.0, total_chunks / max(chunks_searched, 1))
        
        return {
            'search_time_ms': int(chunks_searched / 100),  # Rough estimate
            'chunks_searched': chunks_searched,
            'vs_full_search': {
                'chunks_skipped': max(0, len(all_files) * 10 - chunks_searched),
                'speedup': f"{speedup:.1f}x",
                'estimated_full_search_ms': len(all_files) * 10,
            },
        }


class DiffAwareSearchService:
    """Public interface for diff-aware search."""
    
    def __init__(self, diff_aware_search: DiffAwareSearch):
        """Initialize service."""
        self.diff_aware_search = diff_aware_search
    
    def search_since_last_edit(
        self,
        buffer_id: str,
        query: str,
        scope: str = "changes+deps",  # "changes" | "changes+deps" | "all"
        include_deleted: bool = False,
        top_k: int = 10,
    ) -> Dict:
        """
        Search only modified files and dependencies.
        
        Args:
            buffer_id: Target buffer
            query: Search query
            scope: Search scope
            include_deleted: Include deleted files?
            top_k: Max results
            
        Returns:
            Search response as dict
        """
        # Map string to enum
        scope_enum = SearchScope(scope)
        
        response = self.diff_aware_search.search_since_last_edit(
            buffer_id,
            query,
            scope_enum,
            include_deleted,
            top_k,
        )
        
        # Convert to dict for JSON serialization
        return {
            "query": response.query,
            "scope_used": response.scope_used.value,
            "files_searched": response.files_searched,
            "files_skipped": response.files_skipped,
            "results": [
                {
                    "file": r.file,
                    "line": r.line,
                    "score": r.score,
                    "snippet": r.snippet,
                    "context": r.context,
                }
                for r in response.results
            ],
            "performance": response.performance,
        }
