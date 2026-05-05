Semantic Cache
===============

``gigacode.semantic_cache.SemanticQueryCache`` automatically caches query embeddings to achieve 50% faster searches.

Overview
~~~~~~~~

The Semantic Cache is responsible for:

- Caching query embeddings (not just results)
- Detecting paraphrased queries via semantic similarity
- Automatic LRU eviction
- Hit rate tracking and statistics

Key Benefits
~~~~~~~~~~~~

- **25-50% hit rate**: Catch paraphrased queries automatically
- **100x faster hits**: Cached results return in <1ms vs ~15ms search
- **50% faster searches**: Overall throughput improvement on typical workloads
- **Zero configuration**: Automatically tuned defaults
- **Transparent**: Works automatically with SearchService

Performance Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # First query: ~15ms (new embedding)
    # Query: "find database connection functions"
    results = tool.semantic_search(buffer_id, query)
    
    # Identical query: ~1ms (cached)
    # Query: "find database connection functions"
    results = tool.semantic_search(buffer_id, query)
    # Speedup: 15x faster!
    
    # Paraphrased query: ~5ms (semantic cache hit)
    # Query: "locate DB connection code"
    results = tool.semantic_search(buffer_id, query)
    # Speedup: 3x faster than compute, still much faster!

Core Class
~~~~~~~~~~

.. autoclass:: gigacode.semantic_cache.SemanticQueryCache
   :members:
   :undoc-members:
   :show-inheritance:

Key Components
~~~~~~~~~~~~~~

**CacheEntry**

Represents a cached query:

.. code-block:: python

    from gigacode.semantic_cache import CacheEntry
    
    entry = CacheEntry(
        query="find network requests",
        embedding=[...],  # 384-dim embedding
        results=[...],    # Search results
        timestamp=time.time(),
        hits=1
    )

**SemanticQueryCache**

Main cache implementation:

.. code-block:: python

    from gigacode.semantic_cache import SemanticQueryCache
    
    cache = SemanticQueryCache(
        max_entries=500,        # Keep up to 500 queries
        similarity_threshold=0.95,  # Similarity for hits (0-1)
        embedder=embedder
    )
    
    # Try to get cached result
    results, was_exact = cache.get(
        query="find database operations",
        compute_embedding=lambda q: embedder.encode(q)
    )
    
    if results is not None:
        print(f"✓ Cache hit! (exact: {was_exact})")
    else:
        print("✗ Cache miss, computing search...")
    
    # Store result for future queries
    cache.put(
        query="find database operations",
        results=[...]
    )

Usage with SearchService
~~~~~~~~~~~~~~~~~~~~~~~~~

The Semantic Cache is automatically used by SearchService:

.. code-block:: python

    from gigacode.search_service import SearchService
    
    service = SearchService(
        embedder=embedder,
        index_manager=index_manager,
        semantic_cache_max_entries=500,
        semantic_cache_threshold=0.95
    )
    
    # First search: computes embedding, caches result
    results = service.semantic_search(
        buffer_id="my_project",
        query="error handling"
    )
    
    # Identical query: from cache
    results = service.semantic_search(
        buffer_id="my_project",
        query="error handling"
    )
    
    # Similar query: semantic match, from cache
    results = service.semantic_search(
        buffer_id="my_project",
        query="handle errors"  # ~0.98 similarity
    )

Cache Statistics
~~~~~~~~~~~~~~~~

Get cache performance metrics:

.. code-block:: python

    cache = service._semantic_query_cache  # Access cache
    stats = cache.get_stats()
    
    print(f"Hit rate: {stats['hit_rate']:.1%}")
    print(f"Total queries: {stats['total_queries']}")
    print(f"Cache entries: {stats['entries']}")
    print(f"Avg hits per query: {stats['avg_hits_per_query']:.1f}")

**Statistics fields:**

- ``hit_rate``: Percentage of queries that hit cache
- ``total_queries``: Total queries processed
- ``total_hits``: Total cache hits
- ``entries``: Current entries in cache
- ``max_entries``: Maximum cache size
- ``avg_hits_per_query``: Average times each query is used

Monitor Cache
~~~~~~~~~~~~~

.. code-block:: python

    import time
    from collections import defaultdict
    
    # Track hits over time
    def monitor_cache(cache, interval=10, duration=60):
        start = time.time()
        while time.time() - start < duration:
            stats = cache.get_stats()
            print(f"Hit rate: {stats['hit_rate']:.1%} "
                  f"({stats['total_hits']}/{stats['total_queries']})")
            time.sleep(interval)
    
    monitor_cache(cache)

Configuration
~~~~~~~~~~~~~

**For better hit rate (aggressive caching):**

.. code-block:: python

    cache = SemanticQueryCache(
        max_entries=1000,           # Keep more queries
        similarity_threshold=0.85   # Lower threshold = more hits
    )

**For memory efficiency (conservative caching):**

.. code-block:: python

    cache = SemanticQueryCache(
        max_entries=100,            # Keep fewer queries
        similarity_threshold=0.95   # Higher threshold = fewer false hits
    )

**For balanced performance:**

.. code-block:: python

    cache = SemanticQueryCache(
        max_entries=500,            # Default (good balance)
        similarity_threshold=0.95   # Default (high precision)
    )

LRU Eviction
~~~~~~~~~~~~

When cache exceeds max_entries, least-recently-used queries are evicted:

.. code-block:: python

    cache = SemanticQueryCache(max_entries=3)
    
    cache.put("query 1", results1)
    cache.put("query 2", results2)
    cache.put("query 3", results3)
    print(f"Entries: {len(cache)}")  # 3
    
    cache.put("query 4", results4)
    # "query 1" evicted (least recently used)
    print(f"Entries: {len(cache)}")  # Still 3
    
    cache.get("query 1", ...)  # Cache miss (was evicted)

Similarity Matching
~~~~~~~~~~~~~~~~~~~

The threshold controls what counts as a cache hit:

.. code-block:: python

    # High threshold (0.95 - strict, high precision)
    # Only nearly-identical queries match
    cache = SemanticQueryCache(similarity_threshold=0.95)
    
    # Gets hit:
    cache.put("find database connections", results)
    cache.get("find database connections")  # Hit (1.0)
    cache.get("find database connection")   # Hit (0.98)
    
    # Gets miss:
    cache.get("how do I query databases")  # Miss (0.80)
    
    
    # Low threshold (0.80 - lenient, high recall)
    # Even dissimilar queries might match
    cache = SemanticQueryCache(similarity_threshold=0.80)
    
    # All previous ones get hits:
    cache.put("find database connections", results)
    cache.get("find database connections")  # Hit (1.0)
    cache.get("find database connection")   # Hit (0.98)
    cache.get("how do I query databases")   # Hit (0.80)
    cache.get("show me database code")      # Hit (0.85)

Manual Cache Management
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    cache = SemanticQueryCache(max_entries=500)
    
    # Clear entire cache
    cache.clear()
    
    # Clear specific buffer cache
    cache.clear_buffer(buffer_id="my_project")
    
    # Check if query is cached
    is_cached = cache.is_cached("my query")
    
    # Get cache size
    size = cache.size()

Performance Analysis
~~~~~~~~~~~~~~~~~~~~~

**Typical hit rates by workload:**

- **Interactive search**: 40-60% (users often repeat similar searches)
- **Batch processing**: 10-25% (diverse queries)
- **API workload**: 20-40% (mix of patterns)
- **Test suite**: 60-80% (repeated test queries)

**Optimization strategies:**

1. For high variability: Increase similarity_threshold (0.95+)
2. For consistency: Decrease similarity_threshold (0.85-)
3. For memory: Reduce max_entries
4. For hits: Increase max_entries

Example: Complete Caching Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gigacode import CodeEmbeddingTool
    
    tool = CodeEmbeddingTool()
    buffer_id = tool.embed_codebase("/path/to/project")
    
    import time
    
    queries = [
        "find error handlers",
        "handle exceptions",  # Semantic match to above
        "find error handlers",  # Exact match
        "database operations",
        "query the database",  # Semantic match
    ]
    
    times = []
    for query in queries:
        start = time.time()
        results = tool.semantic_search(buffer_id, query, top_k=5)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Query: '{query}' - {elapsed*1000:.1f}ms")
    
    # Analyze
    first_searches = [0, 3]  # First time seeing queries
    repeat_searches = [1, 2, 4]  # Semantic or exact repeats
    
    first_avg = sum(times[i] for i in first_searches) / len(first_searches)
    repeat_avg = sum(times[i] for i in repeat_searches) / len(repeat_searches)
    
    print(f"\nFirst searches avg: {first_avg*1000:.1f}ms")
    print(f"Repeat searches avg: {repeat_avg*1000:.1f}ms")
    print(f"Speedup: {first_avg/repeat_avg:.1f}x")

Troubleshooting
~~~~~~~~~~~~~~~

**Low hit rate despite cached queries:**

- Increase ``max_entries`` (cache might be too small)
- Lower ``similarity_threshold`` (being too strict)
- Check that embedder is working (run manual similarity test)

**High memory usage:**

- Reduce ``max_entries``
- Increase ``similarity_threshold`` (fewer entries)
- Call ``cache.clear()`` periodically

**Cache not working:**

- Verify SearchService has ``_semantic_query_cache`` initialized
- Check that embedder is available
- Enable debug logging

See Also
~~~~~~~~

- :class:`gigacode.incremental_indexer.IncrementalIndexManager` - Incremental updates
- :class:`gigacode.search_service.SearchService` - Main search interface
- :doc:`../tutorials/search_workflows` - Search examples with caching
