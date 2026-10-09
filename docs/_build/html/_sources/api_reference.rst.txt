API Reference
==============

Complete API documentation for GigaCode modules and tools.

The main entry point is :class:`gigacode.gigacode_tool.CodeEmbeddingTool`; see
:doc:`api/tools` for its full API. The pages below cover the manager, service,
and utility layers.

.. toctree::
   :maxdepth: 2

   api/tools
   api/managers
   api/services
   api/hybrid_search
   api/duplicate_detector
   api/language_detect
   api/chunker
   api/embedder
   api/metadata_store
   api/incremental_indexer
   api/semantic_cache
   api/faiss_optimizer

Quick Reference
~~~~~~~~~~~~~~~

**Most Used Classes:**

- :class:`gigacode.gigacode_tool.CodeEmbeddingTool` - Main entry point
- :class:`gigacode.index_manager.IndexManager` - Index management
- :class:`gigacode.search_service.SearchService` - Search operations
- :class:`gigacode.buffer_manager.BufferManager` - Buffer management

**Most Used Methods:**

- :meth:`gigacode.gigacode_tool.CodeEmbeddingTool.embed_codebase` - Embed a project
- :meth:`gigacode.gigacode_tool.CodeEmbeddingTool.semantic_search` - Search by meaning
- :meth:`gigacode.gigacode_tool.CodeEmbeddingTool.hybrid_search` - Semantic + lexical search
- :meth:`gigacode.gigacode_tool.CodeEmbeddingTool.write_code` - Update code
- :meth:`gigacode.gigacode_tool.CodeEmbeddingTool.commit` - Commit changes
- :meth:`gigacode.gigacode_tool.CodeEmbeddingTool.find_duplicates` - Find similar code

**Configuration:**

- See :class:`gigacode.operation_config.OperationConfig` for all settings
- Defaults suitable for most projects
- Tune for your specific needs

See Also
~~~~~~~~

- :doc:`tutorials/basic_embed` - Embedding tutorial
- :doc:`tutorials/search_workflows` - Search examples
- :doc:`tutorials/edit_workflows` - Editing tutorial
- :doc:`performance_tuning` - Optimization guide
