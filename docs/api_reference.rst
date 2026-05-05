API Reference
==============

Complete API documentation for GigaCode modules and tools.

Core Tool
~~~~~~~~~

.. automodule:: gigacode.gigacode_tool
   :members:
   :undoc-members:
   :show-inheritance:

Managers
~~~~~~~~

.. toctree::
   :maxdepth: 2

   api/index_manager
   api/buffer_manager
   api/search_service

Services
~~~~~~~~

.. toctree::
   :maxdepth: 2

   api/hybrid_search
   api/duplicate_detector
   api/language_detect

Phase 3 Optimizations
~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   api/incremental_indexer
   api/semantic_cache
   api/faiss_optimizer

Utilities
~~~~~~~~~

.. toctree::
   :maxdepth: 2

   api/chunker
   api/embedder
   api/metadata_store

Quick Reference
~~~~~~~~~~~~~~~

**Most Used Classes:**

- :class:`gigacode.gigacode_tool.CodeEmbeddingTool` - Main entry point
- :class:`gigacode.index_manager.IndexManager` - Index management
- :class:`gigacode.search_service.SearchService` - Search operations
- :class:`gigacode.buffer_manager.BufferManager` - Buffer management

**Most Used Methods:**

- :meth:`CodeEmbeddingTool.embed_codebase` - Embed a project
- :meth:`CodeEmbeddingTool.semantic_search` - Search by meaning
- :meth:`CodeEmbeddingTool.hybrid_search` - Semantic + lexical search
- :meth:`CodeEmbeddingTool.write_code` - Update code
- :meth:`CodeEmbeddingTool.commit` - Commit changes
- :meth:`CodeEmbeddingTool.find_duplicates` - Find similar code

**Configuration:**

- See :class:`gigacode.operation_config.OperationConfig` for all settings
- Defaults suitable for most projects
- Tune for your specific needs

See Also
~~~~~~~~

- :doc:`../tutorials/basic_embed` - Embedding tutorial
- :doc:`../tutorials/search_workflows` - Search examples
- :doc:`../tutorials/edit_workflows` - Editing tutorial
- :doc:`../performance_tuning` - Optimization guide
