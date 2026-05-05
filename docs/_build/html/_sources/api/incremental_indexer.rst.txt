Incremental Indexer
===================

``gigacode.incremental_indexer.IncrementalIndexManager`` enables 5-50x faster code commits by only re-embedding changed chunks.

Overview
~~~~~~~~

The Incremental Indexer is responsible for:

- Detecting chunk-level changes in files
- Caching unchanged chunk embeddings
- Re-embedding only changed chunks
- Merging old and new embeddings
- Providing efficiency metrics

Key Benefits
~~~~~~~~~~~~

- **5-50x faster commits**: Only changed chunks are re-embedded
- **Intelligent caching**: Embeddings reused for unchanged chunks
- **Transparent integration**: Works automatically with IndexManager
- **Cost reduction**: Fewer embedding computations = less GPU usage

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Scenario: 2,500 chunks, 10% changed

    Standard re-indexing:
    - Re-embed: 2,500 chunks
    - Time: ~30 seconds
    - Embeddings computed: 2,500

    With incremental indexing:
    - Re-embed: 250 chunks (10%)
    - Reuse: 2,250 chunks (90%)
    - Time: ~3 seconds
    - Embeddings computed: 250
    - Speedup: 10x faster!

Core Class
~~~~~~~~~~

.. autoclass:: gigacode.incremental_indexer.IncrementalIndexManager
   :members:
   :undoc-members:
   :show-inheritance:

Key Components
~~~~~~~~~~~~~~

**ChunkDiffTracker**

Tracks which chunks have changed:

.. code-block:: python

    from gigacode.incremental_indexer import ChunkDiffTracker
    
    tracker = ChunkDiffTracker()
    
    # Register chunks
    tracker.register_chunk(
        chunk_id="chunk_1",
        text="def my_function(): pass",
        file_path="module.py",
        line_start=10,
        line_end=11
    )
    
    # Detect changes
    changed, removed, kept = tracker.detect_changes(
        new_chunks=[
            {
                "id": "chunk_1",
                "text": "def my_function_renamed(): pass"  # Changed!
            }
        ]
    )
    
    print(f"Changed: {len(changed)}")
    print(f"Removed: {len(removed)}")
    print(f"Kept: {len(kept)}")

**IncrementalIndexManager**

Manages incremental embeddings:

.. code-block:: python

    from gigacode.incremental_indexer import IncrementalIndexManager
    
    manager = IncrementalIndexManager(embedder=embedder)
    
    # Compute incremental update
    result = manager.compute_incremental_update(
        buffer_id="my_project",
        new_chunks=[...],
        previous_state={...}
    )
    
    print(f"Changed chunks: {result['changed_count']}")
    print(f"Reused embeddings: {result['reused_count']}")
    print(f"Efficiency: {result['efficiency_ratio']:.1f}x faster")

Usage with IndexManager
~~~~~~~~~~~~~~~~~~~~~~~~

The Incremental Indexer is automatically used by IndexManager:

.. code-block:: python

    from gigacode.index_manager import IndexManager
    
    manager = IndexManager(embedder=embedder)
    
    # When files change and rebuild is triggered:
    # 1. Chunk diff is detected
    # 2. Only changed chunks are embedded
    # 3. Old embeddings are reused
    # 4. Index is updated with merged embeddings
    # 5. Efficiency metrics are logged

Example: Commit with Incremental Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gigacode import CodeEmbeddingTool
    import time
    
    tool = CodeEmbeddingTool()
    buffer_id = tool.embed_codebase("/path/to/project")
    
    # Make small changes (1% of chunks)
    file_path = "/path/to/project/module.py"
    content = open(file_path).read()
    new_content = content.replace("old_name", "new_name")
    
    # Write change
    tool.write_code(buffer_id, "module.py", new_content)
    
    # Commit (incremental indexing activates)
    start = time.time()
    result = tool.commit(buffer_id, ["module.py"], "Rename function")
    elapsed = time.time() - start
    
    print(f"Commit time: {elapsed:.3f}s")
    print(f"Changed chunks: {result['chunks_changed']}")
    print(f"Reused chunks: {result['chunks_kept']}")
    if result.get('efficiency_ratio'):
        print(f"Efficiency: {result['efficiency_ratio']:.1f}x faster")

Configuration
~~~~~~~~~~~~~

.. code-block:: python

    from gigacode.incremental_indexer import IncrementalIndexManager
    
    manager = IncrementalIndexManager(
        embedder=embedder,
        max_cache_size=10000,  # Cache up to 10k embeddings
        use_gpu=True,  # GPU for diff computation
        fallback_threshold=0.5  # Fallback to full rebuild if >50% changed
    )

Statistics
~~~~~~~~~~

Get incremental indexing statistics:

.. code-block:: python

    stats = manager.get_stats()
    print(f"Total updates: {stats['total_updates']}")
    print(f"Average efficiency: {stats['avg_efficiency']:.1f}x")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"Fallback to full rebuild: {stats['fallback_count']} times")

Optimization Tips
~~~~~~~~~~~~~~~~~

**1. Batch changes**

Commit multiple files together for better efficiency:

.. code-block:: python

    # Good: single commit
    tool.commit(buffer_id, ["file1.py", "file2.py", "file3.py"])
    
    # Bad: multiple commits (each rebuilds index)
    tool.commit(buffer_id, ["file1.py"])
    tool.commit(buffer_id, ["file2.py"])
    tool.commit(buffer_id, ["file3.py"])

**2. Monitor efficiency**

.. code-block:: python

    result = tool.commit(buffer_id, files)
    
    # Low efficiency means most chunks changed
    if result['efficiency_ratio'] < 2.0:
        print("Warning: Most code changed, consider full rebuild")

**3. Clear cache periodically**

.. code-block:: python

    # Cache can grow over time
    manager.clear_cache()  # Or specific buffer: clear_cache(buffer_id)

Troubleshooting
~~~~~~~~~~~~~~~

**Commit falling back to full rebuild:**

- Check log for messages about high change percentage
- If >50% of chunks changed, full rebuild is more efficient
- Consider if you're embedding the right files

**Out of memory during incremental update:**

- Reduce batch size (fewer files per commit)
- Clear old cache entries: ``manager.clear_cache()``
- Check ``max_cache_size`` setting

**Incremental updates not working:**

- Verify IndexManager has ``_incremental_manager`` initialized
- Check that file paths match exactly
- Enable debug logging to see what's happening

Advanced: Manual Incremental Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can manually compute incremental updates:

.. code-block:: python

    from gigacode.incremental_indexer import IncrementalIndexManager, ChunkDiffTracker
    
    tracker = ChunkDiffTracker()
    manager = IncrementalIndexManager(embedder=embedder)
    
    # Register baseline chunks
    for chunk in original_chunks:
        tracker.register_chunk(
            chunk_id=chunk['id'],
            text=chunk['text'],
            file_path=chunk['file'],
            line_start=chunk['start'],
            line_end=chunk['end']
        )
    
    # Detect changes
    changed, removed, kept = tracker.detect_changes(new_chunks)
    
    # Compute update
    result = manager.compute_incremental_update(
        buffer_id="my_project",
        changed_chunks=changed,
        new_chunks=new_chunks,
        old_embeddings=old_embeddings
    )
    
    print(f"Efficiency: {result['efficiency_ratio']:.1f}x")

See Also
~~~~~~~~

- :class:`gigacode.semantic_cache.SemanticQueryCache` - Query caching
- :class:`gigacode.faiss_optimizer.FAISSIndexOptimizer` - Index optimization
- :doc:`../tutorials/edit_workflows` - Editing tutorial with incremental updates
