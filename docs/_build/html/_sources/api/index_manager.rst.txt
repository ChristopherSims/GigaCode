Index Manager
=============

``gigacode.index_manager.IndexManager`` handles FAISS and BM25 index caching and management.

Overview
~~~~~~~~

The IndexManager is responsible for:

- Creating and maintaining FAISS indices for semantic search
- Creating and maintaining BM25 indices for lexical search
- Caching indices in memory with an LRU bound
- Rebuilding indices incrementally after edits
- Tracking cache statistics and health

Core Class
~~~~~~~~~~

.. autoclass:: gigacode.index_manager.IndexManager
   :members: create_indices, get_cache_stats, health_check, close
   :undoc-members:
   :show-inheritance:

Create and cache the semantic and lexical indices for a buffer:

.. code-block:: python

    result = manager.create_indices(
        buffer_id="my_project",
        embeddings=embeddings,
        chunks=chunks,
    )

Incremental updates
~~~~~~~~~~~~~~~~~~~

After edits, ``_rebuild_files`` re-chunks the changed files and patches the
index only for those files:

.. code-block:: python

    manager._rebuild_files(buffer_id="my_project", files=["src/app.py"])

See Also
~~~~~~~~

- :class:`gigacode.incremental_indexer.IncrementalIndexManager` - Incremental updates
- :class:`gigacode.faiss_optimizer.FAISSIndexOptimizer` - Index optimization
- :class:`gigacode.gpu_index.GpuIndex` - GPU-accelerated indices
