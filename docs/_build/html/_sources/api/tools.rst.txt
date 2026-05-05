Tools
=====

Main tools and API interfaces for GigaCode.

CodeEmbeddingTool
~~~~~~~~~~~~~~~~~

The primary tool for all GigaCode operations.

.. automodule:: gigacode.gigacode_tool
   :members: CodeEmbeddingTool
   :undoc-members:
   :show-inheritance:

Key Methods
~~~~~~~~~~~

**Embedding:**

- :meth:`embed_codebase` - Embed a codebase
- :meth:`embed_codebase_streaming` - Stream large projects

**Search:**

- :meth:`semantic_search` - Search by meaning
- :meth:`lexical_search` - Search by keywords
- :meth:`hybrid_search` - Combined search

**Editing:**

- :meth:`write_code` - Modify code
- :meth:`commit` - Save and index changes

**Analysis:**

- :meth:`find_duplicates` - Find duplicate code
- :meth:`cluster_code` - Cluster similar code

**Management:**

- :meth:`get_buffer_metadata` - Get buffer info
- :meth:`list_buffers` - List all buffers
- :meth:`delete_buffer` - Delete a buffer

Example
~~~~~~~

.. code-block:: python

    from gigacode import CodeEmbeddingTool
    
    # Initialize
    tool = CodeEmbeddingTool()
    
    # Embed codebase
    buffer_id = tool.embed_codebase("/path/to/project")
    
    # Search
    results = tool.semantic_search(
        buffer_id=buffer_id,
        query="find database functions",
        top_k=5
    )
    
    # Edit and commit
    tool.write_code(buffer_id, "file.py", new_content)
    tool.commit(buffer_id, ["file.py"], "Update file")

Configuration
~~~~~~~~~~~~~

Initialize with custom settings:

.. code-block:: python

    tool = CodeEmbeddingTool(
        work_dir="/path/to/work",
        max_buffers=10,
        enable_gpu=True,
        embedding_model="all-MiniLM-L6-v2"
    )

See Also
~~~~~~~~

- :class:`gigacode.search_service.SearchService` - Search implementation
- :class:`gigacode.buffer_manager.BufferManager` - Buffer management
- :class:`gigacode.index_manager.IndexManager` - Index management
