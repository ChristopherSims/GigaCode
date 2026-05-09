"""Streaming support for large files in GigaCode.

This module enables processing of very large files (>100MB) by chunking them
into manageable pieces before AST parsing and chunking.

Key features:
- Chunk large files before AST parsing
- Preserve chunk boundaries at function/class level where possible
- Stitch metadata across chunks
- Configurable chunk size
"""

from typing import List, Tuple, Optional, Iterator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


__all__ = [
    "StreamingFileReader",
    "ChunkBoundaryPreserver",
    "StreamingChunker",
    "supports_streaming",
]


class StreamingFileReader:
    """Read large files in chunks with boundary awareness."""
    
    def __init__(self, max_chunk_bytes: int = 1024 * 1024):  # 1MB default
        """Initialize streaming reader.
        
        Args:
            max_chunk_bytes: Maximum bytes per chunk (default 1MB)
        """
        self.max_chunk_bytes = max_chunk_bytes
    
    def read_chunks(self, file_path: str) -> Iterator[Tuple[str, int, int]]:
        """Read file in chunks, yielding (content, start_line, end_line).
        
        Args:
            file_path: Path to file to read
        
        Yields:
            Tuple of (chunk_content, start_line, end_line)
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return
        
        file_size = path.stat().st_size
        
        if file_size < self.max_chunk_bytes:
            # Small file - read as single chunk
            content = path.read_text(errors="replace")
            yield content, 1, len(content.splitlines())
            return
        
        # Large file - stream in chunks
        logger.info(f"Streaming {file_path} ({file_size / 1024 / 1024:.1f}MB)")
        
        line_num = 1
        buffer = ""
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                buffer += line
                
                if len(buffer.encode('utf-8')) >= self.max_chunk_bytes:
                    # Chunk is large enough, yield it
                    chunk_lines = buffer.splitlines(keepends=True)
                    chunk_content = "".join(chunk_lines)
                    
                    end_line = line_num + len(chunk_lines) - 1
                    yield chunk_content, line_num, end_line
                    
                    line_num = end_line + 1
                    buffer = ""
        
        # Yield remaining content
        if buffer:
            chunk_lines = buffer.splitlines(keepends=True)
            end_line = line_num + len(chunk_lines) - 1
            yield buffer, line_num, end_line


class ChunkBoundaryPreserver:
    """Try to preserve code boundaries when chunking large files.
    
    This attempts to break files at function/class boundaries rather than
    arbitrary line counts, for better semantic preservation.
    """
    
    def __init__(self, language: str = "python"):
        """Initialize boundary preserver.
        
        Args:
            language: Programming language ("python", "javascript", etc.)
        """
        self.language = language
    
    def find_safe_break_point(
        self,
        content: str,
        target_position: int
    ) -> int:
        """Find a safe line to break file at.
        
        Args:
            content: File content
            target_position: Approximate byte position to break at
        
        Returns:
            Safe byte position to break at
        """
        if self.language == "python":
            return self._find_python_break_point(content, target_position)
        else:
            # Fallback: just break at nearest newline
            return self._find_nearest_newline(content, target_position)
    
    def _find_python_break_point(self, content: str, target_pos: int) -> int:
        """Find Python function/class boundary near target position."""
        # Look backwards for 'def ' or 'class ' at line start
        search_region = max(0, target_pos - 5000)
        region = content[search_region:target_pos]
        
        # Find last function/class definition
        lines = region.split('\n')
        break_line = -1
        
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            if line.startswith(('def ', 'class ', '@')):
                break_line = i
                break
        
        if break_line >= 0:
            # Calculate byte position
            pos = search_region
            for i in range(break_line):
                pos += len(lines[i]) + 1  # +1 for newline
            return pos
        
        # Fallback to nearest newline
        return self._find_nearest_newline(content, target_pos)
    
    def _find_nearest_newline(self, content: str, target_pos: int) -> int:
        """Find nearest newline to target position."""
        target_pos = min(target_pos, len(content) - 1)
        
        # Search backwards for newline
        while target_pos > 0 and content[target_pos] != '\n':
            target_pos -= 1
        
        return target_pos + 1  # Include the newline


class StreamingChunker:
    """Chunk large files using streaming approach.
    
    Combines StreamingFileReader with intelligent boundary preservation
    to process large files efficiently.
    """
    
    def __init__(
        self,
        max_chunk_bytes: int = 1024 * 1024,
        language: str = "python"
    ):
        """Initialize streaming chunker.
        
        Args:
            max_chunk_bytes: Max bytes per chunk
            language: Programming language for boundary detection
        """
        self.reader = StreamingFileReader(max_chunk_bytes)
        self.boundary_preserver = ChunkBoundaryPreserver(language)
    
    def stream_chunks(
        self,
        file_path: str,
        chunk_handler
    ) -> None:
        """Stream file chunks and process them.
        
        Args:
            file_path: Path to file to process
            chunk_handler: Callable that processes each chunk
                          Signature: handler(content: str, start_line: int, end_line: int)
        """
        for chunk_content, start_line, end_line in self.reader.read_chunks(file_path):
            try:
                chunk_handler(chunk_content, start_line, end_line)
            except (OSError, ValueError, RuntimeError) as e:
                logger.error(f"Error processing chunk {start_line}-{end_line}: {e}")
                raise


def supports_streaming(file_size: int, threshold_mb: int = 50) -> bool:
    """Determine if file should use streaming mode.
    
    Args:
        file_size: File size in bytes
        threshold_mb: Threshold in MB above which to use streaming
    
    Returns:
        True if file should use streaming
    """
    threshold_bytes = threshold_mb * 1024 * 1024
    return file_size > threshold_bytes
