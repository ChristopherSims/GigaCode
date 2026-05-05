"""Integration guide for streaming support in GigaCode.

This module documents how to integrate StreamingChunker into existing
BufferManager and SearchService components.

Integration patterns:
1. BufferManager: Use streaming when embedding large files
2. SearchService: Use streaming for index building on large files
3. Incremental indexing: Update index chunk-by-chunk without full re-embedding
"""

from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class StreamingIntegrationGuide:
    """Reference implementation for streaming integration."""
    
    @staticmethod
    def integrate_streaming_into_buffer_manager() -> str:
        """Show how to integrate streaming into BufferManager.embed_file()."""
        return """
# Integration Pattern 1: BufferManager.embed_file()

from gigacode.streaming_support import StreamingChunker, supports_streaming
from pathlib import Path

# In BufferManager.embed_file():

def embed_file(self, file_path: str, language: str = None) -> Dict:
    path = Path(file_path)
    file_size = path.stat().st_size
    
    # Check if file needs streaming
    if supports_streaming(file_size, threshold_mb=50):
        logger.info(f"Using streaming for {file_size / 1024 / 1024:.1f}MB file")
        return self._embed_file_streaming(file_path, language)
    else:
        # Existing non-streaming path
        return self._embed_file_regular(file_path, language)

def _embed_file_streaming(self, file_path: str, language: str = None) -> Dict:
    chunker = StreamingChunker(max_chunk_bytes=1024*1024, language=language)
    all_chunks = []
    metadata = {}
    
    def process_chunk(content: str, start_line: int, end_line: int):
        # Parse and chunk as usual
        tree_sitter_chunks = self.chunker.parse_and_chunk(
            content,
            file_path,
            language,
            line_offset=start_line - 1
        )
        all_chunks.extend(tree_sitter_chunks)
        metadata[f"chunk_{start_line}_{end_line}"] = {
            "line_range": (start_line, end_line),
            "chunk_count": len(tree_sitter_chunks)
        }
    
    chunker.stream_chunks(file_path, process_chunk)
    
    # Embed all chunks as batch
    embeddings = self.embedder.embed_batch(
        [c.content for c in all_chunks]
    )
    
    return {
        "file_path": file_path,
        "chunks": all_chunks,
        "embeddings": embeddings,
        "metadata": metadata,
        "total_chunks": len(all_chunks)
    }
"""
    
    @staticmethod
    def integrate_streaming_into_search_service() -> str:
        """Show how to integrate streaming into SearchService index building."""
        return """
# Integration Pattern 2: SearchService.build_index()

def build_index_streaming(
    self,
    file_paths: List[str],
    incremental: bool = True
) -> Dict:
    '''Build index with streaming support for large files.'''
    
    index_data = {
        "embeddings": [],
        "metadata": [],
        "chunks": []
    }
    
    for file_path in file_paths:
        file_size = Path(file_path).stat().st_size
        
        if supports_streaming(file_size):
            # Use streaming for large files
            logger.info(f"Building index with streaming: {file_path}")
            chunk_data = self._index_file_streaming(file_path)
        else:
            # Regular indexing for small files
            chunk_data = self._index_file_regular(file_path)
        
        index_data["embeddings"].extend(chunk_data["embeddings"])
        index_data["metadata"].extend(chunk_data["metadata"])
        index_data["chunks"].extend(chunk_data["chunks"])
    
    # Build FAISS index from all embeddings
    index = faiss.IndexFlatL2(index_data["embeddings"].shape[1])
    index.add(index_data["embeddings"])
    
    return {
        "index": index,
        "metadata": index_data["metadata"],
        "chunks": index_data["chunks"]
    }

def _index_file_streaming(self, file_path: str) -> Dict:
    '''Index single large file with streaming.'''
    from gigacode.streaming_support import StreamingChunker
    
    chunker = StreamingChunker(max_chunk_bytes=1024*1024)
    all_embeddings = []
    all_metadata = []
    
    def process_chunk(content: str, start_line: int, end_line: int):
        # Parse and embed chunk
        chunks = self.chunker.parse_and_chunk(content, file_path)
        embeddings = self.embedder.embed_batch(
            [c.content for c in chunks]
        )
        
        all_embeddings.extend(embeddings)
        for chunk in chunks:
            chunk.line_start += start_line - 1  # Adjust line numbers
            all_metadata.append({
                "file": file_path,
                "lines": (chunk.line_start, chunk.line_end),
                "chunk_type": chunk.type
            })
    
    chunker.stream_chunks(file_path, process_chunk)
    
    return {
        "embeddings": np.array(all_embeddings),
        "metadata": all_metadata
    }
"""
    
    @staticmethod
    def implement_incremental_indexing() -> str:
        """Show how to implement incremental indexing with streaming."""
        return """
# Integration Pattern 3: Incremental Index Updates

class IncrementalIndexer:
    '''Update index incrementally without full re-embedding.'''
    
    def update_index_streaming(
        self,
        index,
        file_path: str,
        previous_chunk_count: int = 0
    ) -> Dict:
        '''Update index with new version of file using streaming.'''
        from gigacode.streaming_support import StreamingChunker
        
        chunker = StreamingChunker(max_chunk_bytes=1024*1024)
        new_embeddings = []
        new_metadata = []
        chunk_index = 0
        
        def process_chunk(content: str, start_line: int, end_line: int):
            nonlocal chunk_index
            
            # Parse and embed new chunk
            chunks = self.chunker.parse_and_chunk(content, file_path)
            embeddings = self.embedder.embed_batch(
                [c.content for c in chunks]
            )
            
            new_embeddings.extend(embeddings)
            for chunk in chunks:
                chunk.line_start += start_line - 1
                new_metadata.append({
                    "file": file_path,
                    "lines": (chunk.line_start, chunk.line_end),
                    "chunk_id": f"{file_path}#{chunk_index}"
                })
                chunk_index += 1
        
        chunker.stream_chunks(file_path, process_chunk)
        
        # Remove old chunks for this file from index
        old_ids = self._find_chunks_for_file(index, file_path)
        if old_ids:
            index.remove_ids(old_ids)
        
        # Add new chunks
        index.add_with_ids(
            np.array(new_embeddings),
            np.arange(index.ntotal, index.ntotal + len(new_embeddings))
        )
        
        return {
            "added": len(new_embeddings),
            "removed": len(old_ids) if old_ids else 0,
            "metadata": new_metadata
        }
"""
    
    @staticmethod
    def add_streaming_detection() -> str:
        """Show how to add automatic streaming detection."""
        return """
# Integration Pattern 4: Automatic Streaming Detection

# In GigaCodeAPI:

async def embed_codebase_auto_streaming(
    self,
    repo_path: str,
    use_streaming: bool = "auto"
) -> EmbeddingResponse:
    '''Embed codebase with automatic streaming detection.'''
    from gigacode.streaming_support import supports_streaming
    from pathlib import Path
    
    files = self._collect_files(repo_path)
    streaming_files = []
    regular_files = []
    
    # Separate files by size
    for file_path in files:
        file_size = Path(file_path).stat().st_size
        
        if use_streaming == "auto" and supports_streaming(file_size):
            streaming_files.append(file_path)
        else:
            regular_files.append(file_path)
    
    if streaming_files:
        logger.info(
            f"Using streaming for {len(streaming_files)} large files, "
            f"regular for {len(regular_files)} small files"
        )
    
    # Process with appropriate strategy
    results = {
        "streamed": await self._embed_files_streaming(streaming_files),
        "regular": await self._embed_files_regular(regular_files)
    }
    
    return EmbeddingResponse(results=results)
"""
    
    @staticmethod
    def memory_monitoring() -> str:
        """Show how to monitor memory during streaming."""
        return """
# Integration Pattern 5: Memory Monitoring

import tracemalloc
from gigacode.streaming_support import StreamingChunker

def embed_with_memory_monitoring(file_path: str) -> Dict:
    '''Embed file with memory monitoring for optimization.'''
    
    tracemalloc.start()
    
    chunker = StreamingChunker(max_chunk_bytes=1024*1024)
    peak_memory = 0
    chunk_memories = []
    
    def process_chunk(content: str, start_line: int, end_line: int):
        nonlocal peak_memory
        
        # Get memory before processing
        _, peak_before = tracemalloc.get_traced_memory()
        
        # Process chunk
        embeddings = embedder.embed_batch(parse_and_chunk(content))
        
        # Get memory after processing
        current, peak = tracemalloc.get_traced_memory()
        peak_memory = max(peak_memory, peak)
        
        chunk_memories.append({
            "lines": (start_line, end_line),
            "memory_used": peak - peak_before,
            "peak_total": peak
        })
    
    chunker.stream_chunks(file_path, process_chunk)
    
    tracemalloc.stop()
    
    return {
        "peak_memory_mb": peak_memory / 1024 / 1024,
        "per_chunk_memory": chunk_memories
    }
"""


def print_integration_guide():
    """Print all integration patterns."""
    guide = StreamingIntegrationGuide()
    
    print("=" * 80)
    print("STREAMING INTEGRATION GUIDE")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("Pattern 1: BufferManager Integration")
    print("=" * 80)
    print(guide.integrate_streaming_into_buffer_manager())
    
    print("\n" + "=" * 80)
    print("Pattern 2: SearchService Integration")
    print("=" * 80)
    print(guide.integrate_streaming_into_search_service())
    
    print("\n" + "=" * 80)
    print("Pattern 3: Incremental Indexing")
    print("=" * 80)
    print(guide.implement_incremental_indexing())
    
    print("\n" + "=" * 80)
    print("Pattern 4: Automatic Streaming Detection")
    print("=" * 80)
    print(guide.add_streaming_detection())
    
    print("\n" + "=" * 80)
    print("Pattern 5: Memory Monitoring")
    print("=" * 80)
    print(guide.memory_monitoring())


if __name__ == "__main__":
    print_integration_guide()
