Basic Embedding Workflow
=========================

This tutorial covers the fundamental workflow: initializing GigaCode, embedding a codebase, and inspecting the results.

Overview
~~~~~~~~

The embedding workflow consists of three main steps:

1. **Initialize** - Create a CodeEmbeddingTool instance
2. **Embed** - Process your codebase to create embeddings
3. **Inspect** - View metadata about your embedded project

Step 1: Initialize GigaCode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, import and initialize the tool:

.. code-block:: python

    from gigacode import CodeEmbeddingTool
    import os
    from pathlib import Path
    
    # Create a work directory for GigaCode
    work_dir = Path.home() / ".gigacode"
    work_dir.mkdir(exist_ok=True)
    
    # Initialize the tool
    tool = CodeEmbeddingTool(
        work_dir=str(work_dir),
        max_buffers=5,  # Maximum concurrent projects
        enable_gpu=True,  # Use GPU if available
    )
    
    print(f"✓ GigaCode initialized at {work_dir}")

**Configuration Options:**

- ``work_dir`` (str): Directory to store embeddings and caches
- ``max_buffers`` (int): Maximum number of concurrent projects (default: 10)
- ``enable_gpu`` (bool): Enable GPU acceleration (default: True if CUDA available)
- ``embedding_model`` (str): Model name for embeddings (default: "all-MiniLM-L6-v2")

Step 2: Embed Your Codebase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now embed a Python project:

.. code-block:: python

    # Path to your project
    project_path = "/path/to/your/project"
    
    # Embed the entire codebase
    buffer_id = tool.embed_codebase(
        path=project_path,
        buffer_id="my_project",
        include_extensions=[".py", ".js", ".ts"],  # Include these files
        exclude_dirs=["node_modules", ".git", "__pycache__"],  # Skip these dirs
    )
    
    print(f"✓ Embedded project: {buffer_id}")

**What happens during embedding:**

1. Files are discovered and filtered by extension
2. Code is chunked into semantic units
3. Each chunk is converted to a vector embedding
4. Embeddings are indexed for fast search
5. Metadata is stored for reconstruction

**Progress feedback:**

For larger projects, you'll see progress:

.. code-block:: python

    # Enable progress reporting
    result = tool.embed_codebase(
        path=project_path,
        buffer_id="large_project",
        verbose=True  # Print progress
    )
    
    # Output:
    # Discovering files... (found 1234 files)
    # Filtering by extension... (1034 Python files)
    # Chunking code... (2547 chunks)
    # Embedding chunks... [████████████░░░░░░░░] 60%
    # Indexing... (FAISS index created)
    # ✓ Complete!

Step 3: Inspect the Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

View information about your embedded project:

.. code-block:: python

    # Get buffer metadata
    metadata = tool.get_buffer_metadata(buffer_id="my_project")
    
    print(f"Project: {metadata['buffer_id']}")
    print(f"Files: {metadata['file_count']}")
    print(f"Chunks: {metadata['chunk_count']}")
    print(f"Embedding dimension: {metadata['embedding_dim']}")
    print(f"Index type: {metadata['index_type']}")
    print(f"Total size: {metadata['total_size_mb']:.2f} MB")

**Example output:**

.. code-block:: text

    Project: my_project
    Files: 145
    Chunks: 2,547
    Embedding dimension: 384
    Index type: FAISS
    Total size: 156.23 MB

List all embedded projects:

.. code-block:: python

    # Get all buffers
    buffers = tool.list_buffers()
    
    for buffer_id, metadata in buffers.items():
        print(f"{buffer_id}: {metadata['file_count']} files, "
              f"{metadata['chunk_count']} chunks")

Advanced: File Selection
~~~~~~~~~~~~~~~~~~~~~~~~

**Include only specific file types:**

.. code-block:: python

    buffer_id = tool.embed_codebase(
        path=project_path,
        buffer_id="python_only",
        include_extensions=[".py"],  # Only Python files
    )

**Exclude directories and patterns:**

.. code-block:: python

    buffer_id = tool.embed_codebase(
        path=project_path,
        buffer_id="src_only",
        include_extensions=[".py", ".ts"],
        exclude_dirs=["test", "build", ".venv", "node_modules"],
        exclude_patterns=["*.test.py", "*.spec.ts"],  # Test files
    )

**Include specific subdirectories:**

.. code-block:: python

    import os
    from pathlib import Path
    
    project_root = Path(project_path)
    src_dirs = [str(project_root / "src"), str(project_root / "lib")]
    
    buffer_id = tool.embed_codebase(
        path=src_dirs,  # Pass list of directories
        buffer_id="src_and_lib",
        include_extensions=[".py"],
    )

Advanced: Streaming Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For very large codebases, use streaming embeddings:

.. code-block:: python

    # Embed in chunks to avoid memory issues
    buffer_id = tool.embed_codebase_streaming(
        path=project_path,
        buffer_id="huge_project",
        batch_size=128,  # Process 128 chunks at a time
        chunk_size=512,  # Characters per chunk
    )

**Streaming vs. Standard:**

- **Standard**: Faster for small-medium projects (<10k files)
- **Streaming**: Better for large projects (>10k files), lower memory
- **Performance**: Both produce identical results, streaming is just more memory-efficient

Advanced: Custom Chunking
~~~~~~~~~~~~~~~~~~~~~~~~~~

Control how code is split into chunks:

.. code-block:: python

    from gigacode.chunker import Chunker
    
    # Create a custom chunker
    chunker = Chunker(
        chunk_size=256,  # Characters per chunk
        overlap=32,  # Overlap between chunks
        language_aware=True,  # Respect language syntax
    )
    
    # Embed with custom chunker
    buffer_id = tool.embed_codebase(
        path=project_path,
        buffer_id="custom_chunked",
        chunker=chunker,
    )

**Chunking strategies:**

- ``chunk_size=256``: Smaller chunks, more precise, more vectors
- ``chunk_size=512``: Balanced approach (recommended)
- ``chunk_size=1024``: Larger chunks, fewer vectors, less precise

Performance Tips
~~~~~~~~~~~~~~~~

**Speed up embedding:**

1. **Use GPU:** Auto-enabled if CUDA available
2. **Filter extensions:** Only include needed files
3. **Exclude large dirs:** Skip ``node_modules``, ``.git``, etc.
4. **Batch operations:** Embed once, search many times
5. **Incremental updates:** Use commits instead of re-embedding

**Reduce memory usage:**

.. code-block:: python

    # Stream large projects
    buffer_id = tool.embed_codebase_streaming(
        path=project_path,
        buffer_id="lean",
        batch_size=64,  # Smaller batch = less memory
    )

**Monitor embedding progress:**

.. code-block:: python

    import time
    
    start = time.time()
    buffer_id = tool.embed_codebase(
        path=project_path,
        buffer_id="timed",
        verbose=True
    )
    elapsed = time.time() - start
    
    metadata = tool.get_buffer_metadata(buffer_id)
    chunks_per_second = metadata['chunk_count'] / elapsed
    print(f"Speed: {chunks_per_second:.0f} chunks/sec")

What's Next?
~~~~~~~~~~~~

Now that you have embeddings, try:

1. **Search for code:** See :doc:`search_workflows` for search examples
2. **Edit and commit:** See :doc:`edit_workflows` for updating your project
3. **Advanced search:** Learn about semantic cache and incremental indexing in :doc:`../api_reference`

Example: Complete Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a complete example combining all the steps:

.. code-block:: python

    from gigacode import CodeEmbeddingTool
    from pathlib import Path
    
    # 1. Initialize
    work_dir = Path.home() / ".gigacode"
    tool = CodeEmbeddingTool(work_dir=str(work_dir), enable_gpu=True)
    
    # 2. Embed a project
    project_path = "/home/user/my_project"
    buffer_id = tool.embed_codebase(
        path=project_path,
        buffer_id="my_project",
        include_extensions=[".py", ".js"],
        exclude_dirs=["venv", "node_modules", ".git"],
        verbose=True
    )
    
    # 3. Inspect results
    metadata = tool.get_buffer_metadata(buffer_id)
    print(f"✓ Embedded {metadata['file_count']} files "
          f"into {metadata['chunk_count']} chunks")
    
    # 4. Ready for search!
    print(f"✓ Ready to search buffer: {buffer_id}")

Troubleshooting
~~~~~~~~~~~~~~~

**Embedding is very slow:**

- First embedding is normal (model download + initial indexing)
- GPU not detected? Check :doc:`../installation`
- Very large codebase? Use streaming embeddings

**Out of memory:**

- Use streaming embeddings: ``embed_codebase_streaming()``
- Reduce ``batch_size`` parameter
- Exclude large directories

**Files not being embedded:**

- Check ``include_extensions`` includes your file types
- Verify ``exclude_dirs`` isn't too broad
- Files might be in ``.gitignore``

**Embedding takes too long:**

- Remove large binary directories from path
- Use ``include_extensions`` to limit file types
- Enable GPU acceleration

Next Step
~~~~~~~~~

Read :doc:`search_workflows` to learn how to search your embeddings.
