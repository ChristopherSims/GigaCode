Performance Tuning
==================

Strategies to optimize GigaCode for your specific workload.

Quick Wins
~~~~~~~~~~

Enable these first for immediate improvements:

**1. Use GPU (2-10x faster)**

.. code-block:: python

    tool = CodeEmbeddingTool(enable_gpu=True)  # Auto-detect CUDA

**2. Use semantic cache (50% faster searches)**

Already enabled by default, but monitor:

.. code-block:: python

    stats = tool.get_cache_stats(buffer_id)
    print(f"Cache hit rate: {stats['hit_rate']:.1%}")

**3. Use incremental indexing (5-50x faster commits)**

Automatic with Phase 3. Verify:

.. code-block:: python

    result = tool.commit(buffer_id, files)
    print(f"Efficiency: {result.get('efficiency_ratio', 1):.1f}x speedup")

Embedding Performance
~~~~~~~~~~~~~~~~~~~~~

**Reduce files to embed:**

.. code-block:: python

    # Only include relevant extensions
    buffer_id = tool.embed_codebase(
        path="/path/to/project",
        include_extensions=[".py", ".js"],  # Skip other types
        exclude_dirs=["venv", "node_modules", ".git"]
    )

**Batch embedding:**

.. code-block:: python

    # Embed once, search many times (good!)
    buffer_id = tool.embed_codebase(path)
    for query in queries:
        results = tool.semantic_search(buffer_id, query)
    
    # Bad: re-embed for each query
    for query in queries:
        buffer_id = tool.embed_codebase(path)  # Slow!

**Stream large projects:**

.. code-block:: python

    buffer_id = tool.embed_codebase_streaming(
        path="/huge/project",
        batch_size=128,  # Process in batches
        chunk_size=512
    )

**Use smaller embedding models (faster, less accurate):**

.. code-block:: python

    tool = CodeEmbeddingTool(
        embedding_model="all-MiniLM-L6-v2"  # Fast (384-dim)
        # vs default "all-mpnet-base-v2" (768-dim, slower)
    )

Search Performance
~~~~~~~~~~~~~~~~~~

**Use filters to narrow search space:**

.. code-block:: python

    # Narrow: 100ms faster
    results = tool.semantic_search(
        buffer_id="huge_project",
        query="database operations",
        top_k=3,
        include_dirs=["src/db"],
        file_patterns=["*.py"],
        threshold=0.85
    )
    
    # Broad: full search (slower)
    results = tool.semantic_search(
        buffer_id="huge_project",
        query="database operations",
        top_k=20,
        threshold=0.0
    )

**Reduce top_k:**

.. code-block:: python

    # Fast: only asking for 3 results
    results = tool.semantic_search(
        buffer_id="my_project",
        query="my query",
        top_k=3  # Low = fast
    )
    
    # Slow: asking for 100 results
    results = tool.semantic_search(
        buffer_id="my_project",
        query="my query",
        top_k=100  # High = slow
    )

**Use caching strategically:**

.. code-block:: python

    # Cache hits are ~100x faster
    # Cache similar queries together:
    
    queries = [
        "database operations",
        "database queries",        # Will hit semantic cache
        "database connections",    # Will hit semantic cache
        "network operations",      # Won't hit cache (different topic)
    ]
    
    for query in queries:
        results = tool.semantic_search(buffer_id, query)

**Lexical search for specific patterns:**

.. code-block:: python

    # Lexical is often faster for specific code:
    results = tool.lexical_search(
        buffer_id="my_project",
        query="def connect",  # Exact keyword search
        top_k=5
    )
    # Much faster than semantic search for this use case!

Commit Performance
~~~~~~~~~~~~~~~~~~

**Batch changes:**

.. code-block:: python

    # Good: single commit
    tool.commit(
        buffer_id,
        files=["file1.py", "file2.py", "file3.py"],
        message="Batch update"
    )
    # Result: 3 seconds total
    
    # Bad: three commits
    tool.commit(buffer_id, ["file1.py"])  # 1.2s
    tool.commit(buffer_id, ["file2.py"])  # 1.2s
    tool.commit(buffer_id, ["file3.py"])  # 1.2s
    # Total: 3.6 seconds (more overhead!)

**Only commit changed files:**

.. code-block:: python

    # Efficient: only changed file
    tool.commit(
        buffer_id,
        files=["changed_file.py"],
        message="Small fix"
    )
    # With incremental indexing: <100ms
    
    # Inefficient: commit unchanged files
    tool.commit(
        buffer_id,
        files=["changed_file.py", "unchanged1.py", "unchanged2.py"],
        message="Update"
    )
    # Slower: re-indexes unchanged files

**Monitor efficiency:**

.. code-block:: python

    result = tool.commit(buffer_id, files)
    print(f"Efficiency: {result.get('efficiency_ratio', 1):.1f}x")
    
    # <2x = most code changed (consider full rebuild)
    # 5-10x = typical incremental speedup
    # >20x = very efficient (minimal changes)

Indexing Performance
~~~~~~~~~~~~~~~~~~~~

**Monitor cache hit rates:**

.. code-block:: python

    stats = tool.get_cache_stats(buffer_id)
    print(f"Cache hits: {stats['hit_rate']:.1%}")
    
    # <10% = cache not helping, consider changing queries
    # 20-40% = good, typical interactive use
    # >50% = excellent, very repetitive workload

**Clear cache if memory is an issue:**

.. code-block:: python

    # Clear all caches
    tool.clear_cache()
    
    # Or specific buffer
    tool.clear_cache(buffer_id)

**Monitor index size:**

.. code-block:: python

    metadata = tool.get_buffer_metadata(buffer_id)
    print(f"Index size: {metadata['total_size_mb']:.1f} MB")
    
    # If too large, consider:
    # - Excluding more directories
    # - Splitting into multiple buffers
    # - Using smaller embedding model

System-Level Tuning
~~~~~~~~~~~~~~~~~~~

**GPU Optimization**

.. code-block:: python

    # Check GPU availability
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Force specific GPU
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
    
    # Enable GPU features
    tool = CodeEmbeddingTool(
        enable_gpu=True,
        gpu_device=0  # Device ID
    )

**CPU Optimization**

.. code-block:: python

    # Use all CPU cores
    import os
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    
    # NumPy will use all cores
    tool = CodeEmbeddingTool(enable_gpu=False)

**Memory Management**

.. code-block:: python

    # Limit buffers
    tool = CodeEmbeddingTool(
        max_buffers=2,  # Max 2 concurrent projects
        max_cache_size=500  # Max cache entries
    )
    
    # Monitor memory
    import psutil
    process = psutil.Process()
    print(f"Memory: {process.memory_info().rss / 1024**2:.0f} MB")

**Disk I/O Optimization**

.. code-block:: python

    # Use fast SSD for work_dir
    tool = CodeEmbeddingTool(
        work_dir="/mnt/nvme/gigacode"  # NVMe SSD
    )
    
    # Avoid network drives
    # Better: local disk > network NFS > cloud storage

Workload-Specific Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Interactive Search (IDE Plugin)**

.. code-block:: python

    # Goal: Fast response time
    
    tool = CodeEmbeddingTool(enable_gpu=True)
    
    results = tool.semantic_search(
        buffer_id="my_project",
        query=query,
        top_k=3,           # Few results needed
        threshold=0.85,    # Skip low scores
        file_patterns=["*.py"]  # Current language
    )

**Batch Analysis**

.. code-block:: python

    # Goal: Throughput (many searches)
    
    # Pre-cache common queries
    common_queries = [...]
    for query in common_queries:
        tool.semantic_search(buffer_id, query)
    
    # Then process batch
    for item in batch:
        results = tool.semantic_search(buffer_id, item.query)

**Large Codebase Analysis**

.. code-block:: python

    # Goal: Handle large projects
    
    # Split into multiple buffers by language
    py_buffer = tool.embed_codebase(
        path="/path/to/project",
        include_extensions=[".py"],
        buffer_id="project_python"
    )
    
    js_buffer = tool.embed_codebase(
        path="/path/to/project",
        include_extensions=[".js", ".ts"],
        buffer_id="project_js"
    )
    
    # Search both: combined = full coverage
    py_results = tool.semantic_search(py_buffer, query)
    js_results = tool.semantic_search(js_buffer, query)

**Real-time Incremental Updates**

.. code-block:: python

    # Goal: Keep index in sync with active development
    
    # Use file watcher
    from gigacode.file_watcher import FileWatcher
    
    watcher = FileWatcher(project_path)
    
    for changed_files in watcher.watch():
        # Write changes
        for file_path in changed_files:
            content = open(file_path).read()
            tool.write_code(buffer_id, file_path, content)
        
        # Commit (incremental, fast)
        tool.commit(buffer_id, changed_files)

Benchmarking Your Setup
~~~~~~~~~~~~~~~~~~~~~~~

Create a benchmark script:

.. code-block:: python

    import time
    from gigacode import CodeEmbeddingTool
    
    tool = CodeEmbeddingTool(enable_gpu=True)
    
    # Benchmark 1: Embedding
    start = time.time()
    buffer_id = tool.embed_codebase("/path/to/project")
    embed_time = time.time() - start
    
    metadata = tool.get_buffer_metadata(buffer_id)
    chunks_per_sec = metadata['chunk_count'] / embed_time
    print(f"Embedding: {chunks_per_sec:.0f} chunks/sec")
    
    # Benchmark 2: Search
    times = []
    for _ in range(100):
        start = time.time()
        tool.semantic_search(buffer_id, "test query", top_k=5)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"Search: {avg_time*1000:.1f}ms avg, "
          f"{1/avg_time:.1f} queries/sec")
    
    # Benchmark 3: Commit
    # Make a small change
    tool.write_code(buffer_id, "test.py", "new content")
    start = time.time()
    tool.commit(buffer_id, ["test.py"])
    commit_time = time.time() - start
    print(f"Commit: {commit_time*1000:.0f}ms")

Common Performance Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Search is slow:**

1. Check cache hit rate (low? adjust threshold)
2. Reduce top_k
3. Add file/directory filters
4. Use GPU: ``enable_gpu=True``

**Commits are slow:**

1. Check efficiency ratio (low? commit fewer files)
2. Enable GPU
3. Batch changes together
4. Check if incremental indexing working

**High memory usage:**

1. Reduce max_buffers
2. Reduce max_cache_size
3. Clear cache periodically
4. Use smaller embedding model

**Embedding takes forever:**

1. Exclude large directories
2. Use fewer file types
3. Use GPU (2-10x faster)
4. Use streaming embeddings

Next Steps
~~~~~~~~~~

- Profile your specific workload
- Run benchmarks to establish baseline
- Apply targeted optimizations
- Monitor metrics over time
- Adjust thresholds based on results
