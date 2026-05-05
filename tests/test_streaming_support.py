"""Tests for streaming support module.

Tests cover:
- File streaming with various sizes
- Chunk boundary preservation
- Integration with existing code
"""

import pytest
from pathlib import Path
import tempfile
from gigacode.streaming_support import (
    StreamingFileReader,
    ChunkBoundaryPreserver,
    StreamingChunker,
    supports_streaming
)


class TestStreamingFileReader:
    """Test StreamingFileReader class."""
    
    def test_small_file_single_chunk(self):
        """Small file should be read as single chunk."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def hello():\n    return 'world'\n")
            f.flush()
            temp_path = f.name
        
        try:
            reader = StreamingFileReader(max_chunk_bytes=10 * 1024)
            chunks = list(reader.read_chunks(temp_path))
            
            assert len(chunks) == 1
            content, start_line, end_line = chunks[0]
            assert "hello" in content
            assert start_line == 1
            assert end_line == 2
        finally:
            Path(temp_path).unlink()
    
    def test_large_file_multiple_chunks(self):
        """Large file should be split into chunks."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write 100KB of data
            for i in range(10000):
                f.write(f"# Line {i}\nprint({i})\n")
            f.flush()
            temp_path = f.name
        
        try:
            reader = StreamingFileReader(max_chunk_bytes=10 * 1024)
            chunks = list(reader.read_chunks(temp_path))
            
            assert len(chunks) > 1
            
            # Verify line numbers are sequential
            for i, (content, start_line, end_line) in enumerate(chunks):
                assert start_line <= end_line
                assert len(content) > 0
        finally:
            Path(temp_path).unlink()
    
    def test_nonexistent_file(self):
        """Nonexistent file should be handled gracefully."""
        reader = StreamingFileReader()
        chunks = list(reader.read_chunks("/nonexistent/file.py"))
        
        assert len(chunks) == 0


class TestChunkBoundaryPreserver:
    """Test ChunkBoundaryPreserver class."""
    
    def test_python_boundary_detection(self):
        """Python boundary preserver should find function boundaries."""
        content = """
def function_one():
    x = 1
    return x

def function_two():
    y = 2
    return y
"""
        preserver = ChunkBoundaryPreserver(language="python")
        
        # Find break point near end of function_one
        break_pos = preserver.find_safe_break_point(content, 60)
        
        # Should break around a function definition
        assert break_pos > 0
        assert break_pos < len(content)
    
    def test_class_boundary_detection(self):
        """Should detect class boundaries."""
        content = """
class MyClass:
    def __init__(self):
        self.x = 1

class AnotherClass:
    def method(self):
        return self.x
"""
        preserver = ChunkBoundaryPreserver(language="python")
        break_pos = preserver.find_safe_break_point(content, 50)
        
        assert break_pos > 0
    
    def test_fallback_to_newline(self):
        """Should fall back to newline if no boundaries found."""
        content = "x = 1\ny = 2\nz = 3\na = 4\n"
        preserver = ChunkBoundaryPreserver(language="unknown")
        
        break_pos = preserver.find_safe_break_point(content, 10)
        assert break_pos > 0
        assert content[break_pos - 1] == '\n' or break_pos == len(content)


class TestStreamingChunker:
    """Test StreamingChunker integration."""
    
    def test_streaming_chunker_initialization(self):
        """Should initialize with default parameters."""
        chunker = StreamingChunker()
        
        assert chunker.reader is not None
        assert chunker.boundary_preserver is not None
    
    def test_stream_chunks_with_handler(self):
        """Should call handler for each chunk."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write enough data for multiple chunks
            for i in range(5000):
                f.write(f"def func_{i}():\n    return {i}\n\n")
            f.flush()
            temp_path = f.name
        
        try:
            chunker = StreamingChunker(max_chunk_bytes=10 * 1024)
            chunks_processed = []
            
            def handler(content, start_line, end_line):
                chunks_processed.append({
                    "start": start_line,
                    "end": end_line,
                    "size": len(content)
                })
            
            chunker.stream_chunks(temp_path, handler)
            
            assert len(chunks_processed) > 0
            
            # Verify line ranges are sequential
            for i in range(len(chunks_processed) - 1):
                assert chunks_processed[i]["end"] < chunks_processed[i + 1]["start"]
        finally:
            Path(temp_path).unlink()
    
    def test_stream_chunks_error_handling(self):
        """Should handle errors in chunk processing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test():\n    pass\n")
            f.flush()
            temp_path = f.name
        
        try:
            chunker = StreamingChunker()
            
            def failing_handler(content, start_line, end_line):
                raise ValueError("Test error")
            
            with pytest.raises(ValueError):
                chunker.stream_chunks(temp_path, failing_handler)
        finally:
            Path(temp_path).unlink()


class TestSupportsStreaming:
    """Test supports_streaming utility."""
    
    def test_small_file_no_streaming(self):
        """Small file should not need streaming."""
        # 10MB file
        assert supports_streaming(10 * 1024 * 1024, threshold_mb=50) is False
    
    def test_large_file_needs_streaming(self):
        """Large file should need streaming."""
        # 100MB file
        assert supports_streaming(100 * 1024 * 1024, threshold_mb=50) is True
    
    def test_custom_threshold(self):
        """Should respect custom threshold."""
        # 100MB file
        file_size = 100 * 1024 * 1024
        
        assert supports_streaming(file_size, threshold_mb=50) is True
        assert supports_streaming(file_size, threshold_mb=200) is False


class TestStreamingIntegration:
    """Integration tests for streaming with real-like data."""
    
    def test_python_file_streaming_realistic(self):
        """Test streaming with realistic Python code."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write realistic Python code
            code = """
import sys
from pathlib import Path

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, a, b):
        self.result = a + b
        return self.result
    
    def subtract(self, a, b):
        self.result = a - b
        return self.result

def main():
    calc = Calculator()
    print(calc.add(5, 3))
    print(calc.subtract(10, 4))

if __name__ == "__main__":
    main()
"""
            # Repeat to make it larger (need enough to exceed 50KB chunk size)
            for _ in range(1000):
                f.write(code)
            f.flush()
            temp_path = f.name
        
        try:
            chunker = StreamingChunker(max_chunk_bytes=50 * 1024)
            total_lines = 0
            chunk_count = 0
            
            def handler(content, start_line, end_line):
                nonlocal total_lines, chunk_count
                chunk_count += 1
                total_lines += (end_line - start_line + 1)
            
            chunker.stream_chunks(temp_path, handler)
            
            assert chunk_count > 1, f"Expected > 1 chunk, got {chunk_count}"
            assert total_lines > 0
        finally:
            Path(temp_path).unlink()
    
    def test_different_languages(self):
        """Should handle different language boundary preservation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            # JavaScript code
            code = """
function add(a, b) {
    return a + b;
}

function multiply(a, b) {
    return a * b;
}
"""
            for _ in range(1000):
                f.write(code)
            f.flush()
            temp_path = f.name
        
        try:
            # Test with JavaScript language
            chunker = StreamingChunker(
                max_chunk_bytes=10 * 1024,
                language="javascript"
            )
            chunks = []
            
            def handler(content, start_line, end_line):
                chunks.append((start_line, end_line))
            
            chunker.stream_chunks(temp_path, handler)
            
            assert len(chunks) > 1
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
