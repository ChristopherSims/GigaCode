Architecture
=============

Overview of GigaCode's architecture and design patterns.

High-Level Architecture
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────┐
    │                    CodeEmbeddingTool (Main API)             │
    └────┬───────────────────────────────────────────────────┬────┘
         │                                                   │
         ├─ embed_codebase()      ──┐                       │
         ├─ semantic_search()      ──┼─► SearchService     │
         ├─ hybrid_search()        ──┤                       │
         ├─ write_code()       ──┐  │   IndexManager       │
         ├─ commit()           ──┼──┴─► BufferManager      │
         └─ find_duplicates()     │     DuplicateDetector  │
                                   │
                                   └─► Core Managers
                                       (Phase 3 Optimizations)
                                       │
                                       ├─ IncrementalIndexManager
                                       ├─ SemanticQueryCache
                                       └─ FAISSIndexOptimizer

Core Components
~~~~~~~~~~~~~~~

**CodeEmbeddingTool** (``gigacode_tool.py``)

- Main entry point for users
- Delegates to managers
- Handles buffer lifecycle

**BufferManager** (``buffer_manager.py``)

- Manages code buffers
- Tracks file state
- Handles storage

**IndexManager** (``index_manager.py``)

- Manages FAISS indices
- Index caching
- Phase 3: Incremental updates

**SearchService** (``search_service.py``)

- Semantic search
- Lexical search
- Hybrid search
- Phase 3: Query caching

**DuplicateDetector** (``duplicate_detector.py``)

- Finds duplicate code
- Clustering similar code

Data Flow
~~~~~~~~~

**Embedding Workflow**

.. code-block:: text

    User Input (codebase path)
         │
         ▼
    BufferManager: Create buffer
         │
         ▼
    Chunker: Split code into chunks
         │
         ▼
    Embedder: Convert chunks to vectors
         │
         ▼
    IndexManager: Create FAISS index
         │
         ▼
    MetadataStore: Save mappings
         │
         ▼
    Return buffer_id

**Search Workflow**

.. code-block:: text

    User Query
         │
         ▼
    SearchService: Check semantic cache (Phase 3)
         │
         ├─ Hit? Return cached results
         │
         └─ Miss? Continue...
              │
              ▼
              IndexManager: Get FAISS index
              │
              ▼
              Embedder: Embed query
              │
              ▼
              FAISS: Search index
              │
              ▼
              MetadataStore: Reconstruct chunks
              │
              ▼
              Cache result (Phase 3)
              │
              ▼
              Return results

**Commit Workflow**

.. code-block:: text

    User: write_code() + commit()
         │
         ▼
    BufferManager: Track changes
         │
         ▼
    Chunker: Create new chunks
         │
         ▼
    Phase 3: IncrementalIndexManager
    - Detect changed chunks (ChunkDiffTracker)
    - Only embed changed chunks
    - Reuse old embeddings
         │
         ▼
    IndexManager: Update FAISS index
         │
         ▼
    MetadataStore: Update mappings
         │
         ▼
    Return commit result with efficiency metrics

Phase 3 Architecture
~~~~~~~~~~~~~~~~~~~~

Three optimization layers integrated into core:

**Layer 1: Incremental Indexing**

.. code-block:: text

    IndexManager._rebuild_files()
         │
         ▼
    Check if _incremental_manager exists
         │
         ├─ Yes: Use incremental
         │   │
         │   ▼
         │   ChunkDiffTracker.detect_changes()
         │   │
         │   ├─ Changed chunks → Embed
         │   ├─ Removed chunks → Remove from index
         │   └─ Kept chunks → Reuse embeddings
         │
         └─ No: Full rebuild (fallback)
         
    Result: 5-50x faster commits

**Layer 2: Semantic Query Cache**

.. code-block:: text

    SearchService.semantic_search()
         │
         ▼
    Check SemanticQueryCache
         │
         ├─ Exact match → Return cached result (1ms)
         ├─ Semantic match → Return cached result (5ms)
         │  (paraphrased queries detected via cosine similarity)
         │
         └─ No match → Compute search (15ms)
              │
              ▼
              Cache result for future queries
    
    Result: 50% overall speedup, 100x for cache hits

**Layer 3: FAISS Optimization**

.. code-block:: text

    Create index based on vector count:
    
    <10k vectors  → Flat (exact search)
    10k-100k      → IVF (partitioned search)
    >100k         → HNSW (approximate search)
    
    Infrastructure ready for future auto-tuning

Design Patterns
~~~~~~~~~~~~~~~

**Manager Delegation**

Tools delegate to managers:

.. code-block:: python

    class CodeEmbeddingTool:
        def semantic_search(self, buffer_id, query):
            return self._search_service.semantic_search(buffer_id, query)

**Thin Wrapper**

Optimizations wrap existing code without breaking API:

.. code-block:: python

    class SemanticQueryCache:
        def get(self, query):
            # Check cache first, then delegate
            if self._has_exact_match(query):
                return self._cache[query_normalized]
            return None  # Miss, caller does computation

**Strategy Pattern**

Index selection based on size:

.. code-block:: python

    def select_index_type(vector_count):
        if vector_count < 10000:
            return "Flat"
        elif vector_count < 100000:
            return "IVF"
        else:
            return "HNSW"

**Cache Eviction**

LRU (Least Recently Used):

.. code-block:: python

    # When cache full
    remove_least_recently_used()
    add_new_entry()

Extensibility Points
~~~~~~~~~~~~~~~~~~~~

**Custom Embedder**

.. code-block:: python

    class CustomEmbedder(Embedder):
        def encode(self, text):
            # Your implementation
            return embeddings

    tool = CodeEmbeddingTool(embedder=CustomEmbedder())

**Custom Chunker**

.. code-block:: python

    class CustomChunker(Chunker):
        def chunk(self, content, language):
            # Your chunking logic
            return chunks

    buffer_id = tool.embed_codebase(
        path, 
        chunker=CustomChunker()
    )

**Custom Index**

.. code-block:: python

    from gigacode.gpu_index import GpuIndex
    
    class CustomIndex(GpuIndex):
        def search(self, query, k):
            # Your search logic
            return results

Dependencies
~~~~~~~~~~~~

**Core Dependencies:**

- **torch ~2.0.0** - ML framework
- **sentence-transformers ~2.2.0** - Embeddings (384-dim)
- **faiss ~1.8.0** - Vector search
- **tree-sitter ~0.22.0** - Syntax-aware chunking
- **fastapi/uvicorn** - API server
- **numpy** - Array operations

**Optional Dependencies:**

- **faiss-gpu ~1.8.0** - GPU acceleration
- **transformers ~4.37.0** - Language models

Performance Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Memory Usage:**

- Per buffer: ~4 bytes × num_vectors × embedding_dim
- Example: 10k vectors × 384 dims = 15 MB
- Cache overhead: ~1 KB per cached query

**Computation:**

- Embedding: ~100-500 chunks/sec (CPU), 1-10k/sec (GPU)
- Search: 1-50ms per query (depends on size)
- Commit: 1-5ms per changed chunk

**Latency:**

- First search: 10-50ms (index load)
- Cached search: <1ms
- Uncached search: 5-50ms
- GPU search: 2-10x faster

Deployment Patterns
~~~~~~~~~~~~~~~~~~~

**Single Machine (Recommended)**

.. code-block:: python

    # Local-only, simple setup
    tool = CodeEmbeddingTool(work_dir="~/.gigacode")

**Docker Container**

.. code-block:: dockerfile

    FROM python:3.11
    RUN pip install gigacode
    CMD ["python", "-m", "gigacode"]

**API Server** (Future)

.. code-block:: python

    from gigacode import GigacodeServer
    server = GigacodeServer(port=8000)

Security Model
~~~~~~~~~~~~~~

**Data Isolation:**

- No network communication
- All data stored locally
- No cloud or external API calls

**File Access:**

- Only reads/writes specified directories
- Respects user permissions
- No privilege escalation

**Secret Protection:**

- No credentials stored
- No authentication required (local only)
- No telemetry

Testing Architecture
~~~~~~~~~~~~~~~~~~~~

**Test Organization:**

.. code-block:: text

    tests/
    ├─ unit/           # Component tests
    ├─ integration/    # Multi-component tests
    ├─ performance/    # Benchmarks
    └─ e2e/           # End-to-end workflows

**Test Coverage:**

- Phase 3 features: 100% coverage (5 core tests)
- Phase 3b integration: 100% coverage (6 integration tests)
- Existing features: >90% coverage

Future Architecture Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Planned Enhancements:**

- Distributed indexing (multiple machines)
- Incremental model updates
- Query expansion and refinement
- Multi-language embedding models
- Real-time index updates
- Approximate semantic matching for speed

**Backward Compatibility:**

- All enhancements use additive patterns
- Existing APIs remain unchanged
- Opt-in for new features

Debugging & Monitoring
~~~~~~~~~~~~~~~~~~~~~~

**Logging:**

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    tool = CodeEmbeddingTool()  # Now logs detailed info

**Metrics:**

.. code-block:: python

    metrics = tool.get_metrics(buffer_id)
    print(f"Search performance: {metrics['avg_search_ms']:.1f}ms")
    print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")

**Health Check:**

.. code-block:: python

    status = tool.get_health_status()
    print(f"GPU available: {status['gpu_available']}")
    print(f"Indices cached: {status['cached_indices']}")

See Also
~~~~~~~~

- :doc:`performance_tuning` - Optimization strategies
- :doc:`api_reference` - API documentation
- :doc:`tutorials/basic_embed` - Usage examples
