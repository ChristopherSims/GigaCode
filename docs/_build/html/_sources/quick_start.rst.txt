Quick Start
===========

Get GigaCode up and running in 5 minutes.

Installation
~~~~~~~~~~~~

**CPU Version (Default):**

.. code-block:: bash

    pip install .

**GPU Version (Requires CUDA 11.8+):**

.. code-block:: bash

    pip install ".[gpu]"

**Development Setup:**

.. code-block:: bash

    pip install ".[dev]"

Hello World
~~~~~~~~~~~

Embed a codebase and run your first search:

.. code-block:: python

    from gigacode import CodeEmbeddingTool

    # Initialize GigaCode (buffers are persisted under work_dir)
    with CodeEmbeddingTool(work_dir="/tmp/gigacode", device="cpu") as tool:
        # Embed a codebase
        result = tool.embed_codebase("/path/to/my/project", pattern="*.py")
        buffer_id = result["buffer_id"]
        print(f"Embedded project: {buffer_id}")

        # Search for code
        results = tool.semantic_search(
            buffer_id=buffer_id,
            query="find the initialization function",
            top_k=5,
        )

        # Display results
        for match in results["matches"]:
            print(f"{match['file']}:{match['start_line']} - {match['name']}")

What's Next?
~~~~~~~~~~~~

- Read the :doc:`tutorials/basic_embed` tutorial for detailed examples
- Explore :doc:`tutorials/search_workflows` for advanced search patterns
- Check :doc:`api_reference` for the complete API documentation
- See :doc:`performance_tuning` for optimization tips

Common Tasks
~~~~~~~~~~~~

**Search for a function:**

.. code-block:: python

    results = tool.semantic_search(
        buffer_id="my_project",
        query="handle database connections",
        top_k=5
    )

**Edit code and commit changes:**

.. code-block:: python

    # Replace line 12 of database.py (buffer only; disk unchanged until commit)
    tool.write_code(
        buffer_id=buffer_id,
        file="database.py",
        start_line=12,
        new_lines=["    pool = create_pool(max_size=10)"],
        end_line=12,
    )
    tool.diff(buffer_id)                 # preview the change
    tool.commit(buffer_id)               # write to disk

**Find duplicate code:**

.. code-block:: python

    duplicates = tool.find_duplicates(
        buffer_id=buffer_id,
        threshold=0.95,
    )

**Cluster similar functions:**

.. note::

   Clustering is not available in the current build; ``cluster_code`` returns
   an explicit ``status: "unavailable"`` result rather than an empty success.

.. code-block:: python

    clusters = tool.cluster_code(buffer_id=buffer_id, threshold=0.75)

Troubleshooting
~~~~~~~~~~~~~~~

**CUDA not detected (GPU version)?**

Check your CUDA installation:

.. code-block:: bash

    python scripts/check_gpu.py

**Search results are slow?**

- First search loads the index (normal)
- Subsequent searches are cached and fast
- GPU significantly speeds up large searches

**Out of memory errors?**

- Reduce ``max_buffers`` in CodeEmbeddingTool
- For large codebases, use streaming embeddings
- Consider CPU-only mode if GPU memory is limited

**Module not found?**

Make sure you installed with ``pip install .`` not just ``python``:

.. code-block:: bash

    cd /path/to/GigaCode
    pip install .

Performance Tips
~~~~~~~~~~~~~~~~

1. **Use GPU:** 2-10x faster search on NVIDIA GPUs
2. **Batch operations:** Embedding multiple files is faster than individual embeds
3. **Semantic cache:** Repeated queries are cached automatically
4. **Incremental updates:** Only changed files are re-embedded during commits

Get Help
~~~~~~~~

- Check the :doc:`api_reference` for API documentation
- Read :doc:`tutorials/search_workflows` for common patterns
- See :doc:`performance_tuning` for optimization strategies
- Review source code: `github.com/your-repo/gigacode <https://github.com/your-repo/gigacode>`_
