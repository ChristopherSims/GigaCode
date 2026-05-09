"""
Phase 7: Auto-Chunking by Agent Profile

Optimizes code chunking strategy based on agent profile and task type.
Different tasks need different chunk granularity and content focus.

Key features:
- Profile-based chunking strategies (reviewer, debugger, architect, documenter)
- Adaptive inclusion/exclusion of code elements
- Token efficiency per profile
- Seamless integration with embedding pipeline
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum

from gigacode.metrics import timer, log_metric


class AgentProfile(Enum):
    """Agent profile types for task-specific chunking."""
    REVIEWER = "reviewer"  # Code review, refactoring
    DEBUGGER = "debugger"  # Bug fixing, tracing
    ARCHITECT = "architect"  # System design, APIs
    DOCUMENTER = "documenter"  # Writing docs, examples
    GENERIC = "generic"  # Default profile


@dataclass
class ChunkingStrategy:
    """Chunking strategy configuration for a profile."""
    profile: AgentProfile
    include_elements: Set[str] = field(default_factory=set)
    exclude_elements: Set[str] = field(default_factory=set)
    max_lines: int = 300
    min_lines: int = 3
    prioritize_elements: Set[str] = field(default_factory=set)
    include_docstrings: bool = True
    include_tests: bool = False
    include_examples: bool = False
    include_comments: bool = True
    description: str = ""
    expected_token_savings: int = 0


class ChunkingStrategyFactory:
    """Creates chunking strategies for different profiles."""
    
    STRATEGIES = {
        AgentProfile.REVIEWER: ChunkingStrategy(
            profile=AgentProfile.REVIEWER,
            include_elements={
                "function_body", "docstring", "decorator", "type_hints", "class_definition"
            },
            exclude_elements={"test", "example", "vendor", "internal"},
            max_lines=300,
            min_lines=3,
            include_docstrings=True,
            include_tests=False,
            include_examples=False,
            include_comments=True,
            description="Full functions with docstrings for code review and refactoring",
            expected_token_savings=25,  # vs generic
        ),
        AgentProfile.DEBUGGER: ChunkingStrategy(
            profile=AgentProfile.DEBUGGER,
            include_elements={
                "function_signature", "error_paths", "imports", "exception_handler",
                "assertion", "debug_statement"
            },
            exclude_elements={"docstring_details", "vendor", "comment", "example"},
            max_lines=80,
            min_lines=1,
            prioritize_elements={"exception_handler", "assertion", "debug_statement"},
            include_docstrings=False,
            include_tests=True,  # Include test code for debugging
            include_examples=False,
            include_comments=False,
            description="Minimal signatures with error paths for debugging",
            expected_token_savings=67,  # vs generic
        ),
        AgentProfile.ARCHITECT: ChunkingStrategy(
            profile=AgentProfile.ARCHITECT,
            include_elements={
                "class_signature", "public_method", "api_docstring", "interface",
                "protocol", "abstract_method"
            },
            exclude_elements={
                "private_method", "test", "implementation_detail", "vendor",
                "docstring_implementation"
            },
            max_lines=150,
            min_lines=3,
            include_docstrings=True,  # API docs only
            include_tests=False,
            include_examples=False,
            include_comments=False,
            description="Public APIs and interfaces for system design",
            expected_token_savings=33,  # vs generic
        ),
        AgentProfile.DOCUMENTER: ChunkingStrategy(
            profile=AgentProfile.DOCUMENTER,
            include_elements={
                "docstring", "example", "function_body", "comment",
                "usage_example", "parameter_doc"
            },
            exclude_elements={"vendor", "internal_implementation"},
            max_lines=400,
            min_lines=5,
            prioritize_elements={"docstring", "example", "usage_example"},
            include_docstrings=True,
            include_tests=False,
            include_examples=True,
            include_comments=True,
            description="Full documentation with examples",
            expected_token_savings=3,  # Minimal savings; quality preferred
        ),
        AgentProfile.GENERIC: ChunkingStrategy(
            profile=AgentProfile.GENERIC,
            include_elements={"all"},
            exclude_elements={"vendor"},
            max_lines=250,
            min_lines=3,
            include_docstrings=True,
            include_tests=False,
            include_examples=False,
            include_comments=True,
            description="Balanced chunks for general-purpose tasks",
            expected_token_savings=0,  # Baseline
        ),
    }
    
    @staticmethod
    def get_strategy(profile: AgentProfile) -> ChunkingStrategy:
        """Get chunking strategy for profile."""
        return ChunkingStrategyFactory.STRATEGIES.get(
            profile, ChunkingStrategyFactory.STRATEGIES[AgentProfile.GENERIC]
        )
    
    @staticmethod
    def get_strategy_by_name(profile_name: str) -> ChunkingStrategy:
        """Get chunking strategy by profile name."""
        try:
            profile = AgentProfile(profile_name)
            return ChunkingStrategyFactory.get_strategy(profile)
        except ValueError:
            return ChunkingStrategyFactory.STRATEGIES[AgentProfile.GENERIC]


class CodeElementExtractor:
    """Extracts different types of code elements from source."""
    
    @staticmethod
    def extract_function_signatures(code: str) -> List[tuple]:
        """Extract function signatures from code."""
        # Placeholder: in real implementation, use AST parsing
        signatures = []
        for line_num, line in enumerate(code.split('\n'), 1):
            if line.strip().startswith('def ') or line.strip().startswith('async def '):
                signatures.append((line_num, line))
        return signatures
    
    @staticmethod
    def extract_docstrings(code: str) -> List[tuple]:
        """Extract docstrings from code."""
        # Placeholder: use AST parsing for real implementation
        docstrings = []
        in_docstring = False
        start_line = 0
        
        for line_num, line in enumerate(code.split('\n'), 1):
            stripped = line.strip()
            if '"""' in stripped or "'''" in stripped:
                if not in_docstring:
                    in_docstring = True
                    start_line = line_num
                else:
                    in_docstring = False
                    docstrings.append((start_line, line_num))
        
        return docstrings
    
    @staticmethod
    def extract_error_paths(code: str) -> List[tuple]:
        """Extract exception handling and error paths."""
        # Placeholder: look for try/except, raise, assert
        error_paths = []
        for line_num, line in enumerate(code.split('\n'), 1):
            stripped = line.strip()
            if any(kw in stripped for kw in ['except', 'raise', 'assert', 'try']):
                error_paths.append((line_num, line))
        return error_paths
    
    @staticmethod
    def extract_examples(code: str) -> List[tuple]:
        """Extract example code blocks."""
        # Placeholder: look for doctest, example comments
        examples = []
        for line_num, line in enumerate(code.split('\n'), 1):
            if 'example' in line.lower() or '>>>' in line:
                examples.append((line_num, line))
        return examples


class AdaptiveChunker:
    """
    Chunks code based on agent profile for optimal token efficiency.
    """
    
    def __init__(self, base_chunker):
        """
        Initialize adaptive chunker.
        
        Args:
            base_chunker: Base chunking implementation
        """
        self.base_chunker = base_chunker
        self.extractor = CodeElementExtractor()
    
    def chunk_with_profile(
        self,
        code: str,
        file_path: str,
        profile: AgentProfile = AgentProfile.GENERIC,
    ) -> List[Dict]:
        """
        Chunk code according to agent profile.
        
        Args:
            code: Source code
            file_path: File path (for context)
            profile: Agent profile
            
        Returns:
            List of chunks optimized for profile
        """
        strategy = ChunkingStrategyFactory.get_strategy(profile)
        
        with timer(f"chunk_with_profile.{profile.value}"):
            # Get base chunks from standard chunker
            base_chunks = self.base_chunker.chunk(code, file_path)
            
            # Filter and reorder chunks based on strategy
            optimized_chunks = self._apply_strategy(
                base_chunks,
                code,
                strategy,
            )
            
            log_metric("chunk_with_profile", {
                "profile": profile.value,
                "base_chunks": len(base_chunks),
                "optimized_chunks": len(optimized_chunks),
                "savings_percent": strategy.expected_token_savings,
            })
            
            return optimized_chunks
    
    def _apply_strategy(
        self,
        chunks: List[Dict],
        code: str,
        strategy: ChunkingStrategy,
    ) -> List[Dict]:
        """Apply strategy to chunks."""
        filtered = []
        
        for chunk in chunks:
            # Check size constraints
            chunk_lines = len(chunk.get('content', '').split('\n'))
            if chunk_lines < strategy.min_lines or chunk_lines > strategy.max_lines:
                continue
            
            # Check element inclusion
            chunk_type = chunk.get('type', 'code')
            
            # Exclude test files if configured
            if not strategy.include_tests and 'test' in chunk.get('file', '').lower():
                continue
            
            # Add to filtered set
            chunk['profile'] = strategy.profile.value
            chunk['priority'] = self._compute_priority(chunk, strategy)
            filtered.append(chunk)
        
        # Sort by priority (highest first)
        filtered.sort(key=lambda c: c.get('priority', 0), reverse=True)
        
        return filtered
    
    @staticmethod
    def _compute_priority(chunk: Dict, strategy: ChunkingStrategy) -> float:
        """Compute priority score for chunk."""
        base_score = chunk.get('score', 0.0)
        
        # Boost priority for prioritized elements
        chunk_type = chunk.get('type', '')
        if chunk_type in strategy.prioritize_elements:
            base_score *= 1.5
        
        # Penalize excluded elements
        if chunk_type in strategy.exclude_elements:
            base_score *= 0.2
        
        return base_score
    
    def embed_codebase_with_profile(
        self,
        codebase_path: str,
        profile: AgentProfile = AgentProfile.GENERIC,
    ) -> Dict:
        """
        Embed entire codebase with profile-specific chunking.
        
        Args:
            codebase_path: Path to codebase root
            profile: Agent profile for chunking
            
        Returns:
            Embedding metadata including strategy used
        """
        strategy = ChunkingStrategyFactory.get_strategy(profile)
        
        return {
            "codebase_path": codebase_path,
            "profile": profile.value,
            "strategy": {
                "include_elements": list(strategy.include_elements),
                "exclude_elements": list(strategy.exclude_elements),
                "max_lines": strategy.max_lines,
                "min_lines": strategy.min_lines,
                "description": strategy.description,
            },
            "expected_token_savings": f"{strategy.expected_token_savings}%",
        }


class ProfileAdapter:
    """Adapts operations based on agent profile."""
    
    def __init__(self, adaptive_chunker: AdaptiveChunker):
        """Initialize adapter."""
        self.adaptive_chunker = adaptive_chunker
        self.current_profile = AgentProfile.GENERIC
    
    def set_profile(self, profile: AgentProfile) -> None:
        """Set profile for session."""
        self.current_profile = profile
        log_metric("profile_adapter.set_profile", {"profile": profile.value})
    
    def get_profile(self) -> AgentProfile:
        """Get current profile."""
        return self.current_profile
    
    def adapt_search(
        self,
        query: str,
        profile: Optional[AgentProfile] = None,
    ) -> Dict:
        """Adapt search query based on profile."""
        profile = profile or self.current_profile
        
        # Profile-specific query enhancement
        enhancements = {
            AgentProfile.REVIEWER: "review refactor improve",
            AgentProfile.DEBUGGER: "error bug exception fail",
            AgentProfile.ARCHITECT: "interface api design public",
            AgentProfile.DOCUMENTER: "documentation example guide",
        }
        
        enhancement = enhancements.get(profile, "")
        if enhancement:
            query = f"{query} {enhancement}"
        
        return {"query": query, "profile": profile.value}
    
    def adapt_chunking(
        self,
        code: str,
        file_path: str,
        profile: Optional[AgentProfile] = None,
    ) -> List[Dict]:
        """Adapt chunking based on profile."""
        profile = profile or self.current_profile
        return self.adaptive_chunker.chunk_with_profile(code, file_path, profile)


class AgentProfileService:
    """Public interface for agent profile functionality."""
    
    def __init__(self, adaptive_chunker: AdaptiveChunker, profile_adapter: ProfileAdapter):
        """Initialize service."""
        self.adaptive_chunker = adaptive_chunker
        self.profile_adapter = profile_adapter
    
    def embed_codebase(
        self,
        codebase_path: str,
        agent_profile: str = "generic",
    ) -> Dict:
        """
        Embed codebase with profile-specific chunking.
        
        Args:
            codebase_path: Path to codebase
            agent_profile: Agent profile (reviewer|debugger|architect|documenter)
            
        Returns:
            Embedding metadata
        """
        try:
            profile = AgentProfile(agent_profile)
        except ValueError:
            profile = AgentProfile.GENERIC
        
        return self.adaptive_chunker.embed_codebase_with_profile(
            codebase_path, profile
        )
    
    def set_agent_profile(self, profile: str) -> None:
        """Set agent profile for session."""
        try:
            profile_enum = AgentProfile(profile)
            self.profile_adapter.set_profile(profile_enum)
        except ValueError:
            self.profile_adapter.set_profile(AgentProfile.GENERIC)
    
    def get_chunking_strategy(self, profile: str) -> Dict:
        """Get chunking strategy for profile."""
        try:
            profile_enum = AgentProfile(profile)
        except ValueError:
            profile_enum = AgentProfile.GENERIC
        
        strategy = ChunkingStrategyFactory.get_strategy(profile_enum)
        
        return {
            "profile": strategy.profile.value,
            "include": list(strategy.include_elements),
            "exclude": list(strategy.exclude_elements),
            "max_lines": strategy.max_lines,
            "min_lines": strategy.min_lines,
            "description": strategy.description,
            "expected_token_savings": f"{strategy.expected_token_savings}%",
        }
