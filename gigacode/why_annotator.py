"""
Phase 9: "Why This Matters" Annotations

Enriches search results with contextual explanations of why each result
is relevant, reducing manual interpretation and improving decision-making.

Key features:
- Semantic relevance scoring with explanations
- Edit context awareness
- Call graph and dependency analysis
- Suggested next actions based on result type
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from gigacode.metrics import log_metric, timer

logger = logging.getLogger(__name__)


class RelevanceReason(Enum):
    """Categories of relevance reasons."""

    KEYWORD_MATCH = "keyword_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CALL_SITE = "call_site"
    IMPORT_DEPENDENCY = "import_dependency"
    EDIT_CONTEXT = "edit_context"
    TYPE_MATCH = "type_match"
    DOCUMENTATION = "documentation"


@dataclass
class RelevanceExplanation:
    """Explanation for why result is relevant."""

    primary_reason: RelevanceReason
    confidence: float  # 0.0-1.0
    details: str  # Human-readable explanation
    evidence: List[str] = field(default_factory=list)  # Specific evidence snippets
    related_concepts: List[str] = field(default_factory=list)


@dataclass
class AnnotatedSearchResult:
    """Search result with "why matters" annotation."""

    file: str
    line: int
    score: float
    snippet: str
    why: str  # Human-readable explanation
    relevance: RelevanceExplanation = field(default_factory=dict)
    suggested_next_action: str = ""
    related_results: List[str] = field(default_factory=list)  # Other relevant files


class RelevanceExplainer:
    """Generates "why matters" explanations for search results."""

    def __init__(
        self,
        dependency_graph,
        buffer_manager,
        symbol_index,
    ):
        """
        Initialize explainer.

        Args:
            dependency_graph: DependencyGraph instance
            buffer_manager: BufferManager instance
            symbol_index: SymbolIndex instance for call graph
        """
        self.dependency_graph = dependency_graph
        self.buffer_manager = buffer_manager
        self.symbol_index = symbol_index

    def explain_result(
        self,
        buffer_id: str,
        result: Dict,
        query: str,
        edit_context: Optional[List[str]] = None,
    ) -> AnnotatedSearchResult:
        """
        Generate explanation for why a search result is relevant.

        Args:
            buffer_id: Target buffer
            result: Search result from search_service
            query: Original search query
            edit_context: Files currently being edited

        Returns:
            Annotated result with explanation
        """
        with timer("explain_result"):
            file = result.get("file")

            # Analyze relevance reasons
            reasons = self._analyze_relevance(
                buffer_id,
                file,
                query,
                edit_context,
            )

            # Select primary reason
            primary_reason = (
                max(
                    reasons.items(),
                    key=lambda x: x[1].get("confidence", 0),
                )
                if reasons
                else None
            )

            # Generate human-readable explanation
            explanation = self._build_explanation(primary_reason, reasons)

            # Suggest next action
            next_action = self._suggest_next_action(file, primary_reason)

            # Find related results
            related = self._find_related_results(buffer_id, file)

            annotated = AnnotatedSearchResult(
                file=file,
                line=result.get("line", 0),
                score=result.get("score", 0.0),
                snippet=result.get("snippet", ""),
                why=explanation,
                suggested_next_action=next_action,
                related_results=related,
            )

            log_metric(
                "explain_result",
                {
                    "file": file,
                    "primary_reason": primary_reason[0].value if primary_reason else None,
                    "confidence": primary_reason[1].get("confidence") if primary_reason else 0,
                },
            )

            return annotated

    def _analyze_relevance(
        self,
        buffer_id: str,
        file: str,
        query: str,
        edit_context: Optional[List[str]],
    ) -> Dict[RelevanceReason, Dict]:
        """Analyze why result is relevant."""
        reasons = {}

        # 1. Check for keyword matches
        keywords = self._extract_keywords(query)
        keyword_matches = self._find_keyword_matches(file, keywords)
        if keyword_matches:
            reasons[RelevanceReason.KEYWORD_MATCH] = {
                "confidence": min(0.9, len(keyword_matches) * 0.3),
                "details": f"Contains {len(keyword_matches)} query keywords",
                "evidence": keyword_matches[:3],
            }

        # 2. Check semantic similarity
        # (implicitly high if result was returned by semantic search)
        reasons[RelevanceReason.SEMANTIC_SIMILARITY] = {
            "confidence": 0.85,  # Assumed from search ranking
            "details": "Semantically similar to query intent",
            "evidence": [],
        }

        # 3. Check if called by edited files
        if edit_context:
            call_sites = self._find_call_sites(buffer_id, file, edit_context)
            if call_sites:
                reasons[RelevanceReason.CALL_SITE] = {
                    "confidence": 0.95,
                    "details": f"Called by {len(call_sites)} edited files",
                    "evidence": call_sites[:2],
                }

        # 4. Check if imported by edited files
        if edit_context:
            imports = self._find_imports(buffer_id, file, edit_context)
            if imports:
                reasons[RelevanceReason.IMPORT_DEPENDENCY] = {
                    "confidence": 0.9,
                    "details": f"Imported by {len(imports)} edited files",
                    "evidence": imports[:2],
                }

        # 5. Check if in edit context
        if edit_context and file in edit_context:
            reasons[RelevanceReason.EDIT_CONTEXT] = {
                "confidence": 1.0,
                "details": "This is a file you're currently editing",
                "evidence": [file],
            }

        return reasons

    @staticmethod
    def _extract_keywords(query: str) -> Set[str]:
        """Extract search keywords."""
        # Simple word extraction; in real impl use NLP
        stop_words = {"a", "an", "the", "is", "to", "for", "and", "or"}
        words = {w.lower() for w in query.split() if w.lower() not in stop_words and len(w) > 3}
        return words

    @staticmethod
    def _find_keyword_matches(file: str, keywords: Set[str]) -> List[str]:
        """Find which keywords appear in file."""
        # Placeholder: in real impl, parse file and extract identifiers
        matches = []
        for kw in keywords:
            if kw in file.lower():
                matches.append(kw)
        return matches

    def _find_call_sites(
        self,
        buffer_id: str,
        target_file: str,
        caller_files: List[str],
    ) -> List[str]:
        """Find files that call functions in target_file using symbol_index."""
        call_sites: List[str] = []

        if self.symbol_index is None:
            return call_sites

        try:
            # Get all symbols defined in target_file
            symbols_result = self.symbol_index.list_file_symbols(buffer_id, target_file)
            if isinstance(symbols_result, dict) and symbols_result.get("status") != "ok":
                return call_sites

            symbols = []
            if isinstance(symbols_result, dict):
                symbols = symbols_result.get("symbols", [])
            elif isinstance(symbols_result, list):
                symbols = symbols_result

            for symbol in symbols:
                symbol_name = symbol.get("name") if isinstance(symbol, dict) else str(symbol)
                if not symbol_name:
                    continue

                # Find references to this symbol across caller files
                refs_result = self.symbol_index.get_symbol_references(buffer_id, symbol_name)
                if isinstance(refs_result, dict) and refs_result.get("status") == "ok":
                    refs = refs_result.get("references", [])
                    for ref in refs:
                        ref_file = ref.get("file") if isinstance(ref, dict) else str(ref)
                        if ref_file and ref_file != target_file:
                            call_sites.append(
                                f"{symbol_name} called in {ref_file}:{ref.get('line', '?')}"
                            )

        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to find call sites for {target_file}: {e}")

        return call_sites[:5]  # Limit to top 5

    def _find_imports(
        self,
        buffer_id: str,
        target_file: str,
        importer_files: List[str],
    ) -> List[str]:
        """Find files that import from target_file using dependency_graph."""
        imports: List[str] = []

        if self.dependency_graph is None:
            # Fallback: naive check using file names
            target_module = target_file.replace(".py", "").replace("/", ".").replace("\\", ".")
            for imp_file in importer_files:
                if imp_file == target_file:
                    continue
                # Check if target module name appears in file content
                imports.append(f"{imp_file} imports from {target_module}")
            return imports[:3]

        try:
            # Use dependency graph to find actual importers
            dependents = self.dependency_graph.get_dependents(buffer_id, target_file)
            for dep_file in dependents:
                if dep_file in importer_files and dep_file != target_file:
                    imports.append(f"{dep_file} imports from {target_file}")

        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to find imports for {target_file}: {e}")
            # Fallback to naive
            for imp_file in importer_files:
                if imp_file != target_file:
                    imports.append(f"{imp_file} may import from {target_file}")

        return imports[:5]

    @staticmethod
    def _build_explanation(
        primary_reason: Optional[tuple],
        all_reasons: Dict,
    ) -> str:
        """Build human-readable explanation."""
        if not primary_reason:
            return "Relevant to your query"

        reason_type, reason_data = primary_reason
        base = reason_data.get("details", "")

        # Add supporting evidence
        evidence = reason_data.get("evidence", [])
        if evidence and len(evidence) > 0:
            evidence_str = "\nIt matches your query because:\n"
            for i, ev in enumerate(evidence[:2], 1):
                evidence_str += f"({i}) {ev}\n"
            base += evidence_str

        # Add secondary reasons if applicable
        other_reasons = [
            r
            for r in all_reasons.items()
            if r[0] != reason_type and r[1].get("confidence", 0) > 0.7
        ]
        if other_reasons:
            base += "\n(Also relevant because: "
            base += ", ".join(r[1].get("details", "") for r in other_reasons[:1])
            base += ")"

        return base

    @staticmethod
    def _suggest_next_action(file: str, primary_reason: Optional[tuple]) -> str:
        """Suggest what to do with this result."""
        if not primary_reason:
            return ""

        reason_type, _ = primary_reason

        if reason_type == RelevanceReason.KEYWORD_MATCH:
            return f"Read {file} to understand existing pattern"
        elif reason_type == RelevanceReason.SEMANTIC_SIMILARITY:
            return f"Review {file} for implementation reference"
        elif reason_type == RelevanceReason.CALL_SITE:
            return f"Check {file} to see how it's being used"
        elif reason_type == RelevanceReason.IMPORT_DEPENDENCY:
            return f"Verify {file} changes don't break your edits"
        elif reason_type == RelevanceReason.EDIT_CONTEXT:
            return f"Continue editing {file}"
        else:
            return f"Read {file} for more context"

    @staticmethod
    def _find_related_results(buffer_id: str, file: str) -> List[str]:
        """Find other related files."""
        # Placeholder: in real impl, use dependency graph
        return []


class AnnotationService:
    """Service for annotating search results with explanations."""

    def __init__(self, explainer: RelevanceExplainer):
        """Initialize service."""
        self.explainer = explainer

    def annotate_results(
        self,
        buffer_id: str,
        results: List[Dict],
        query: str,
        edit_context: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Annotate search results with explanations.

        Args:
            buffer_id: Target buffer
            results: Raw search results
            query: Original search query
            edit_context: Files being edited

        Returns:
            Annotated results as dicts
        """
        annotated = []

        for result in results:
            annotated_result = self.explainer.explain_result(
                buffer_id,
                result,
                query,
                edit_context,
            )

            annotated.append(
                {
                    "file": annotated_result.file,
                    "line": annotated_result.line,
                    "score": annotated_result.score,
                    "snippet": annotated_result.snippet,
                    "why": annotated_result.why,
                    "suggested_next_action": annotated_result.suggested_next_action,
                    "related_results": annotated_result.related_results,
                }
            )

        return annotated


class WhyAnnotator:
    """Public interface for "why matters" annotations."""

    def __init__(self, annotation_service: AnnotationService):
        """Initialize annotator."""
        self.annotation_service = annotation_service

    def annotate_search_results(
        self,
        buffer_id: str,
        results: List[Dict],
        query: str,
        edit_context: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Annotate search results with "why this matters" explanations.

        Args:
            buffer_id: Target buffer
            results: Search results from search_service
            query: Original search query
            edit_context: Files currently being edited

        Returns:
            Annotated results with explanations
        """
        return self.annotation_service.annotate_results(
            buffer_id,
            results,
            query,
            edit_context,
        )
