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

    # Initialize (buffers persist under work_dir)
    with CodeEmbeddingTool(work_dir="/path/to/work", device="cpu") as tool:
        # Embed a codebase
        result = tool.embed_codebase("/path/to/project", pattern="*.py")
        buffer_id = result["buffer_id"]

        # Search
        results = tool.semantic_search(
            buffer_id=buffer_id,
            query="find database functions",
            top_k=5,
        )

        # Edit (buffer only) and commit to disk
        tool.write_code(
            buffer_id=buffer_id,
            file="file.py",
            start_line=1,
            new_lines=["def connect():\n"],
            end_line=1,
        )
        tool.commit(buffer_id)

Configuration
~~~~~~~~~~~~~

Initialize with custom settings:

.. code-block:: python

    tool = CodeEmbeddingTool(
        work_dir="/path/to/work",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_gpu=True,
        max_buffers=10,
        tool_profile="editing",  # expose write/commit tools to server transports
    )

See Also
~~~~~~~~

- :class:`gigacode.search_service.SearchService` - Search implementation
- :class:`gigacode.buffer_manager.BufferManager` - Buffer management
- :class:`gigacode.index_manager.IndexManager` - Index management
