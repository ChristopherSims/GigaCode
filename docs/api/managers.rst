Managers
========

Core manager classes for buffer, index, and search management.

Overview
~~~~~~~~

Managers handle the core functionality:

- **BufferManager**: Code buffer lifecycle and file tracking
- **IndexManager**: FAISS index creation and caching
- **SearchService**: All search operations

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   buffer_manager
   index_manager
   search_service

Quick Reference
~~~~~~~~~~~~~~~

**BufferManager:**

- Create and manage buffers
- Track file changes
- Handle storage

**IndexManager:**

- Create FAISS indices
- Cache indices efficiently
- Auto-select index types (Phase 3)
- Support incremental updates (Phase 3)

**SearchService:**

- Semantic search
- Lexical search
- Hybrid search
- Query caching (Phase 3)

See Also
~~~~~~~~

- :doc:`tools` - CodeEmbeddingTool (main API)
- :doc:`../tutorials/basic_embed` - Embedding examples
- :doc:`../tutorials/search_workflows` - Search examples
- :doc:`../tutorials/edit_workflows` - Edit examples
