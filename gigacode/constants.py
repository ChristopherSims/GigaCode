"""Centralized constants for GigaCode.

All magic numbers and default configuration values live here.
"""

from __future__ import annotations

# Embedding defaults
DEFAULT_BATCH_SIZE = 64
DEFAULT_BATCH_THRESHOLD = 100

# Search defaults
DEFAULT_TOP_K = 5
MAX_TOP_K = 10_000
MAX_MAX_RESULTS = 100_000
SIMILARITY_THRESHOLD = 0.95
# Max chars of match source text echoed back in search results.
# Full source must be fetched via read_code.
MAX_MATCH_TEXT_PREVIEW_CHARS = 200

# Buffer / cache defaults
DEFAULT_MAX_BUFFERS = 10
DEFAULT_QUERY_CACHE_SIZE = 256
DEFAULT_SEMANTIC_QUERY_CACHE_SIZE = 500
MAX_DIRTY_BEFORE_AUTO_REBUILD = 3

# Size / resource limits
DEFAULT_THRESHOLD_MB = 500.0
STREAMING_THRESHOLD_MB = 50

# Server / metrics defaults
DEFAULT_PROMETHEUS_PORT = 9090
DEFAULT_HTTP_PORT = 8765

# Query / rate limits
DEFAULT_RATE_LIMIT_PER_MINUTE = 60
MAX_QUERY_LENGTH = 10_000

# Agent-facing response shaping
# read_code returns at most this many lines per call when end_line is omitted
READ_CODE_DEFAULT_WINDOW = 150
# write_code responses cap the echoed unified diff at this many lines
WRITE_DIFF_MAX_LINES = 60
# Repeat invocations of an identical search (tool + canonical args) get a hint
REPEAT_HINT_AFTER_CALLS = 2
# Below this similarity the client is nudged toward hybrid/literal search
WEAK_SEMANTIC_SCORE = 0.2

# Chunking defaults
DEFAULT_SLIDING_WINDOW_SIZE = 30

# Token estimation
CHARS_PER_TOKEN = 4.0
LINES_PER_TOKEN_ESTIMATE = 8
