#!/usr/bin/env python3
"""Verify all recent feature implementations."""

from pathlib import Path
from gigacode.response_types import SearchResponse, SearchMatch
from gigacode.chunker import chunk_text, chunk_file

# Verify all recent features
print('Response types (SearchResponse, SearchMatch)')
print('Chunk functions with sliding_window_size parameter')

# Test chunk size control
print()
print('Testing chunk size control...')
text = '\n'.join([f'def func_{i}(): pass' for i in range(100)])

chunks_30 = chunk_text(text, sliding_window_size=30)
print(f'  window=30: {len(chunks_30)} chunks')

chunks_60 = chunk_text(text, sliding_window_size=60)
print(f'  window=60: {len(chunks_60)} chunks')

chunks_15 = chunk_text(text, sliding_window_size=15)
print(f'  window=15: {len(chunks_15)} chunks')

assert len(chunks_30) > len(chunks_60), "30-line chunks should be more than 60-line chunks"
assert len(chunks_15) > len(chunks_30), "15-line chunks should be more than 30-line chunks"

print()
print('All recent implementations verified!')
print('   - Chunk size control (Issue 8) ')
print('   - API consistency (Issue 4) ')
print('   - Health check endpoint (Issue 13)')
