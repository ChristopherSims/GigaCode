Index Manager
=============

``gigacode.index_manager.IndexManager`` handles FAISS and BM25 index caching and management.

Overview
~~~~~~~~

The IndexManager is responsible for:

- Creating and maintaining FAISS indices for semantic search
- Creating and maintaining BM25 indices for lexical search
- Automatic index selection based on buffer size
- Caching indices in memory
- Garbage collection of unused indices

Core Class
~~~~~~~~~~

.. autoclass:: gigacode.index_manager.IndexManager
   :members:
   :undoc-members:
   :show-inheritance:

Key Methods
~~~~~~~~~~~

**Index Creation**

.. automethod:: gigacode.index_manager.IndexManager.get_or_create_faiss_index
   :noindex:

Create or retrieve a FAISS index for a buffer:

.. code-block:: python

    index = manager.get_or_create_faiss_index(
        buffer_id="my_project",
        embeddings=embeddings,
        embedding_dim=384
    )

**Index Retrieval**

.. automethod:: gigacode.index_manager.IndexManager.get_faiss_index
   :noindex:

Retrieve an existing FAISS index:

.. code-block:: python

    index = manager.get_faiss_index(buffer_id="my_project")
    if index is None:
        print("Index not found")

**Index Invalidation**

.. automethod:: gigacode.index_manager.IndexManager.invalidate_index
   :noindex:

Clear a cached index:

.. code-block:: python

    manager.invalidate_index(buffer_id="my_project")

**Garbage Collection**

.. automethod:: gigacode.index_manager.IndexManager.cleanup_indices
   :noindex:

Clean up unused indices:

.. code-block:: python

    manager.cleanup_indices()

Phase 3 Integration
~~~~~~~~~~~~~~~~~~~

The IndexManager automatically uses :class:`gigacode.incremental_indexer.IncrementalIndexManager` for efficient updates:

- When files change, only changed chunks are re-embedded
- Embeddings are cached and reused
- 5-50x faster commits compared to full re-indexing

Example: Incremental Update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gigacode.index_manager import IndexManager
    from gigacode.incremental_indexer import IncrementalIndexManager
    
    manager = IndexManager(embedder=embedder)
    
    # Files are changed
    # Manager automatically:
    # 1. Detects which chunks changed
    # 2. Only re-embeds changed chunks
    # 3. Reuses embeddings for unchanged chunks
    # 4. Updates index with new embeddings

Configuration
~~~~~~~~~~~~~

.. code-block:: python

    from gigacode.index_manager import IndexManager
    
    manager = IndexManager(
        embedder=embedder,
        max_indices=10,  # Max concurrent indices
        auto_gc=True,  # Auto cleanup unused
        gc_interval=3600  # Cleanup every hour
    )

See Also
~~~~~~~~

- :class:`gigacode.incremental_indexer.IncrementalIndexManager` - Incremental updates
- :class:`gigacode.faiss_optimizer.FAISSIndexOptimizer` - Index optimization
- :class:`gigacode.gpu_index.GpuIndex` - GPU-accelerated indices
