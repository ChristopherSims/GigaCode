Search Service
===============

``gigacode.search_service.SearchService`` handles all search operations.

Overview
~~~~~~~~

The SearchService is responsible for:

- Semantic search using embeddings
- Lexical search using BM25
- Hybrid search combining both
- Result ranking and filtering
- Query caching for performance

Core Class
~~~~~~~~~~

.. autoclass:: gigacode.search_service.SearchService
   :members:
   :undoc-members:
   :show-inheritance:

Search Methods
~~~~~~~~~~~~~~

**Semantic Search**

Search by meaning and intent:

.. code-block:: python

    results = service.semantic_search(
        buffer_id="my_project",
        query="database connection",
        top_k=5,
        threshold=0.8
    )

Returns matches ranked by semantic similarity (0-1 score).

**Lexical Search**

Search by exact keywords:

.. code-block:: python

    results = service.lexical_search(
        buffer_id="my_project",
        query="def connect",
        top_k=5
    )

Returns matches containing the keywords.

**Hybrid Search**

Combine semantic and lexical:

.. code-block:: python

    results = service.hybrid_search(
        buffer_id="my_project",
        query="database operations",
        top_k=5,
        semantic_weight=0.6,
        lexical_weight=0.4
    )

Returns best matches from both methods, weighted and combined.

Result Format
~~~~~~~~~~~~~

All search methods return results with this structure:

.. code-block:: python

    {
        "matches": [
            {
                "file": "src/database.py",
                "start_line": 42,
                "end_line": 58,
                "name": "connect",
                "type": "function",
                "score": 0.92,
                "text": "def connect(...):\n    ...",
            },
            # ... more matches
        ],
        "total": 1,
        "took_ms": 12.3
    }

**Result Fields:**

- ``file``: Relative file path
- ``start_line``: Starting line number (1-indexed)
- ``end_line``: Ending line number
- ``name``: Symbol name (function, class, etc.)
- ``type``: Symbol type (function, class, variable, etc.)
- ``score``: Relevance score (0-1)
- ``text``: Full code snippet
- ``chunk_id``: Internal chunk identifier

Advanced Features
~~~~~~~~~~~~~~~~~

**Semantic Cache**

Phase 3 integration automatically caches query embeddings:

.. code-block:: python

    # First search: computes embedding (~15ms)
    results1 = service.semantic_search(
        buffer_id="my_project",
        query="find database connections"
    )
    
    # Identical query: from cache (~1ms)
    results2 = service.semantic_search(
        buffer_id="my_project",
        query="find database connections"
    )
    
    # Similar query: semantic cache hit (~5ms)
    results3 = service.semantic_search(
        buffer_id="my_project",
        query="locate database connection code"
    )

Cache hit rate typically 25-50% on real workloads.

**View cache statistics:**

.. code-block:: python

    stats = service.get_cache_stats(buffer_id="my_project")
    print(f"Hit rate: {stats['hit_rate']:.1%}")
    print(f"Entries: {stats['entries']}")

**Filtering Results**

.. code-block:: python

    results = service.semantic_search(
        buffer_id="my_project",
        query="handle errors",
        top_k=20,
        file_patterns=["*.py"],  # Only .py files
        languages=["python"]  # Only Python
    )
    
    # Filter by score
    high_conf = [m for m in results["matches"] if m["score"] > 0.85]

Configuration
~~~~~~~~~~~~~

.. code-block:: python

    from gigacode.search_service import SearchService
    
    service = SearchService(
        embedder=embedder,
        index_manager=index_manager,
        semantic_cache_max_entries=500,
        semantic_cache_threshold=0.95,
        use_gpu=True
    )

Performance Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Semantic Search:**

- First search: 10-50ms (index load)
- Cached queries: <1ms
- GPU enabled: 2-10x faster
- Large index (>100k vectors): may use IVF approximation

**Lexical Search:**

- Small project (<10k chunks): 1-5ms
- Medium project (10k-100k chunks): 5-50ms
- Large project (>100k chunks): 50-500ms

**Hybrid Search:**

- Combines both methods
- Slower than individual methods
- Best quality results

Tuning Guide
~~~~~~~~~~~~

**For fast results:**

.. code-block:: python

    # Use semantic cache + GPU + high threshold
    results = service.semantic_search(
        buffer_id="my_project",
        query="my query",
        threshold=0.90  # Higher = fewer results, faster filtering
    )

**For accurate results:**

.. code-block:: python

    # Hybrid search, lower threshold, semantic-focused
    results = service.hybrid_search(
        buffer_id="my_project",
        query="my query",
        threshold=0.70,  # Lower = more results
        semantic_weight=0.8,
        lexical_weight=0.2
    )

**For code patterns:**

.. code-block:: python

    # Lexical search with regex
    results = service.lexical_search(
        buffer_id="my_project",
        query=r"def\s+\w+\s*\(.*\):",  # Regex pattern
        use_regex=True
    )

See Also
~~~~~~~~

- :class:`gigacode.semantic_cache.SemanticQueryCache` - Query caching
- :class:`gigacode.hybrid_search.HybridSearcher` - Hybrid implementation
- :doc:`../tutorials/search_workflows` - Search examples
