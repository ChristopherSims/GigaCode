Edit Workflows
==============

This tutorial covers updating your code and efficiently committing changes with incremental indexing.

Overview
~~~~~~~~

The edit workflow consists of:

1. **Write** - Modify code in your buffer
2. **Commit** - Save and re-index your changes
3. **Verify** - Confirm changes are indexed correctly

With Phase 3 optimizations, commits are 5-50x faster by only re-embedding changed chunks.

Prerequisites
~~~~~~~~~~~~~

You need an embedded project. If not, see :doc:`basic_embed`:

.. code-block:: python

    from gigacode import CodeEmbeddingTool
    
    tool = CodeEmbeddingTool()
    buffer_id = tool.embed_codebase("/path/to/project")

Step 1: Write Code
~~~~~~~~~~~~~~~~~~

Update code in your buffer:

.. code-block:: python

    # Read existing code
    existing_code = open("/path/to/project/example.py").read()
    
    # Modify code
    new_code = existing_code.replace(
        "old_function_name",
        "new_function_name"
    )
    
    # Write modified code to buffer
    tool.write_code(
        buffer_id="my_project",
        file="example.py",
        content=new_code
    )

**Key parameters:**

- ``buffer_id`` (str): Your project identifier
- ``file`` (str): File path relative to project root
- ``content`` (str): New file content

**Write multiple files:**

.. code-block:: python

    files_to_update = {
        "module1.py": new_content1,
        "utils/helpers.py": new_content2,
        "config/settings.json": new_content3
    }
    
    for file_path, content in files_to_update.items():
        tool.write_code(
            buffer_id="my_project",
            file=file_path,
            content=content
        )

**Read and modify:**

.. code-block:: python

    from pathlib import Path
    
    # Read current code
    file_path = Path("/path/to/project") / "example.py"
    content = file_path.read_text()
    
    # Modify code (example: add docstring)
    new_content = add_docstring(content)
    
    # Write back to buffer
    tool.write_code(
        buffer_id="my_project",
        file="example.py",
        content=new_content
    )

Step 2: Commit Changes
~~~~~~~~~~~~~~~~~~~~~~

Save and re-index your changes:

.. code-block:: python

    # Commit single file
    result = tool.commit(
        buffer_id="my_project",
        files=["example.py"],
        message="Update function names for clarity"
    )
    
    print(f"✓ Committed {result['files_updated']} files")
    print(f"✓ Re-indexed {result['chunks_updated']} chunks")
    print(f"✓ Took {result['time_ms']:.0f}ms")

**Commit multiple files:**

.. code-block:: python

    result = tool.commit(
        buffer_id="my_project",
        files=["module1.py", "utils/helpers.py", "config/settings.json"],
        message="Refactor: improve code organization"
    )

**Commit entire project:**

.. code-block:: python

    result = tool.commit(
        buffer_id="my_project",
        files=None,  # None means all files
        message="Major refactor: update architecture"
    )

**Commit with metadata:**

.. code-block:: python

    result = tool.commit(
        buffer_id="my_project",
        files=["feature.py"],
        message="Add new feature: caching",
        author="Your Name",
        timestamp=datetime.now().isoformat()
    )

Phase 3 Optimization: Incremental Indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GigaCode automatically uses incremental indexing, which is 5-50x faster:

.. code-block:: python

    # Before commit (Phase 3 optimization active)
    start = time.time()
    
    result = tool.commit(
        buffer_id="my_project",
        files=["example.py"],  # Only changed file
        message="Small fix"
    )
    
    elapsed = time.time() - start
    print(f"Commit time: {elapsed:.3f}s (with incremental indexing)")
    print(f"Chunks changed: {result['chunks_changed']}")
    print(f"Chunks kept (cached): {result['chunks_kept']}")
    print(f"Efficiency: {result['efficiency_ratio']:.1f}x faster")

**Performance comparison:**

.. code-block:: text

    Scenario: 1% of 2,500 chunks changed

    Without incremental indexing:
    - Re-embed: 2,500 chunks
    - Time: ~30 seconds

    With incremental indexing (Phase 3):
    - Re-embed: 25 chunks (changed)
    - Reuse: 2,475 chunks (unchanged)
    - Time: ~0.15 seconds
    - Speedup: 200x faster!
```

**How incremental indexing works:**

1. **Detect changes**: Compare file hash with stored hash
2. **Chunk diff**: Find which chunks changed
3. **Selective re-embed**: Only embed changed chunks
4. **Reuse embeddings**: Keep unchanged chunk embeddings
5. **Update index**: Merge new and old embeddings

**Configure incremental indexing:**

.. code-block:: python

    tool = CodeEmbeddingTool(
        incremental_indexing=True,  # Enable (default)
        incremental_threshold=0.1,  # Fallback to full rebuild if >10% changed
    )

Advanced: Batch Commits
~~~~~~~~~~~~~~~~~~~~~~~

Efficiently process multiple changes:

.. code-block:: python

    import time
    
    # Track changes
    changes = [
        ("module1.py", new_content1),
        ("module2.py", new_content2),
        ("utils.py", new_content3),
    ]
    
    # Apply changes
    for file_path, content in changes:
        tool.write_code(buffer_id="my_project", file=file_path, content=content)
    
    # Single commit for all changes (more efficient)
    result = tool.commit(
        buffer_id="my_project",
        files=[f[0] for f in changes],
        message="Batch update: multiple modules"
    )
    
    print(f"✓ Committed {result['files_updated']} files")
    print(f"✓ {result['chunks_changed']} chunks changed")
    print(f"✓ Took {result['time_ms']:.0f}ms")

Advanced: Commit with Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verify changes after commit:

.. code-block:: python

    # Commit changes
    result = tool.commit(
        buffer_id="my_project",
        files=["module.py"],
        message="Add new function"
    )
    
    # Verify by searching for the new code
    new_results = tool.semantic_search(
        buffer_id="my_project",
        query="the new function",
        top_k=1
    )
    
    if new_results["total"] > 0:
        print("✓ New function is searchable")
        print(f"  Found: {new_results['matches'][0]['name']}")
    else:
        print("✗ New function not indexed (verify commit succeeded)")

Advanced: Rollback Commits
~~~~~~~~~~~~~~~~~~~~~~~~~~

Revert to a previous version:

.. code-block:: python

    # Get commit history
    history = tool.get_commit_history(buffer_id="my_project")
    
    for commit in history:
        print(f"{commit['id']}: {commit['message']} "
              f"({commit['timestamp']})")
    
    # Rollback to previous commit
    tool.rollback_commit(
        buffer_id="my_project",
        commit_id=history[1]['id']
    )
    
    print("✓ Rolled back to previous version")

Advanced: Diff Before Commit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See what changed before committing:

.. code-block:: python

    # Get diff of changes
    diff = tool.get_diff(
        buffer_id="my_project",
        file="example.py"
    )
    
    print("Changes:")
    for line in diff:
        if line.startswith("+"):
            print(f"  Added: {line}")
        elif line.startswith("-"):
            print(f"  Removed: {line}")
        else:
            print(f"  Context: {line}")

Complete Workflow Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a full edit workflow:

.. code-block:: python

    from gigacode import CodeEmbeddingTool
    import time
    from pathlib import Path
    
    # Initialize
    tool = CodeEmbeddingTool()
    buffer_id = tool.embed_codebase("/path/to/project")
    
    # Step 1: Make changes
    project_root = Path("/path/to/project")
    
    # Read original
    file1 = project_root / "module1.py"
    content1 = file1.read_text()
    
    # Modify
    updated_content1 = content1.replace(
        "old_function",
        "new_function"
    )
    
    # Step 2: Write to buffer
    tool.write_code(
        buffer_id=buffer_id,
        file="module1.py",
        content=updated_content1
    )
    
    # Step 3: Commit with timing
    start = time.time()
    result = tool.commit(
        buffer_id=buffer_id,
        files=["module1.py"],
        message="Rename: old_function -> new_function"
    )
    elapsed = time.time() - start
    
    # Step 4: Display results
    print(f"✓ Committed successfully")
    print(f"  Files: {result['files_updated']}")
    print(f"  Chunks changed: {result['chunks_changed']}")
    print(f"  Chunks kept: {result['chunks_kept']}")
    print(f"  Time: {elapsed:.3f}s")
    if result.get('efficiency_ratio'):
        print(f"  Efficiency: {result['efficiency_ratio']:.1f}x faster")
    
    # Step 5: Verify by searching
    results = tool.semantic_search(
        buffer_id=buffer_id,
        query="new function",
        top_k=1
    )
    
    if results["total"] > 0:
        print(f"\n✓ New function indexed: {results['matches'][0]['name']}")

Performance Tips
~~~~~~~~~~~~~~~~

**Speed up commits:**

1. **Batch changes**: Commit multiple files at once
2. **Incremental indexing**: Automatic, 5-50x faster
3. **Only commit changed files**: Skip unchanged files
4. **Use semantic cache**: Next search is instant

**Commit time estimates:**

- Small change (1-5% changed): <100ms (incremental)
- Medium change (10-20% changed): 0.5-1s (incremental)
- Large change (50%+ changed): 2-5s (incremental)
- Full rebuild: 10-30s (only if incremental fails)

**Monitor commit performance:**

.. code-block:: python

    import time
    
    changes_percent = [1, 5, 10, 25, 50]
    
    for pct in changes_percent:
        # Simulate percent changes
        start = time.time()
        result = tool.commit(
            buffer_id=buffer_id,
            files=get_files_to_change(pct),
            message=f"Test {pct}% change"
        )
        elapsed = time.time() - start
        
        print(f"{pct}% changed: {elapsed:.3f}s "
              f"({result.get('efficiency_ratio', 1):.1f}x speedup)")

Common Patterns
~~~~~~~~~~~~~~~

**Pattern 1: Modify and commit**

.. code-block:: python

    # Read -> Modify -> Write -> Commit
    content = (Path(project_root) / file).read_text()
    modified = apply_transformation(content)
    tool.write_code(buffer_id, file, modified)
    tool.commit(buffer_id, [file], "Applied transformation")

**Pattern 2: Bulk update**

.. code-block:: python

    # Update multiple files
    for file_path in files_to_update:
        content = (Path(project_root) / file_path).read_text()
        updated = update_function(content)
        tool.write_code(buffer_id, file_path, updated)
    
    tool.commit(buffer_id, files_to_update, "Bulk update")

**Pattern 3: Auto-save on interval**

.. code-block:: python

    import threading
    
    def auto_commit_worker(tool, buffer_id, interval_seconds=60):
        while True:
            time.sleep(interval_seconds)
            tool.commit(buffer_id, None, "Auto-commit")
    
    # Start background worker
    thread = threading.Thread(
        target=auto_commit_worker,
        args=(tool, buffer_id),
        daemon=True
    )
    thread.start()

**Pattern 4: Track changes**

.. code-block:: python

    # Before
    before = tool.get_buffer_metadata(buffer_id)
    
    # Make changes and commit
    tool.write_code(buffer_id, "file.py", new_content)
    tool.commit(buffer_id, ["file.py"], "Update")
    
    # After
    after = tool.get_buffer_metadata(buffer_id)
    
    # Show diff
    print(f"Chunks before: {before['chunk_count']}")
    print(f"Chunks after: {after['chunk_count']}")
    print(f"Change: {after['chunk_count'] - before['chunk_count']:+d}")

Troubleshooting
~~~~~~~~~~~~~~~

**Commit failed:**

- Check write_code succeeded first
- Ensure file exists in original project
- Try committing entire buffer (files=None)

**Commit is slow:**

- Falling back to full rebuild (check logs)
- Try committing fewer files
- Check disk space available

**New code not searchable:**

- Run commit to finalize changes
- Search results may be cached (try different query)
- Verify with get_buffer_metadata that chunks increased

**Out of memory during commit:**

- Reduce number of files committed at once
- Check incremental_threshold setting
- Use streaming mode for large buffers

Next Steps
~~~~~~~~~~

- Learn search techniques: :doc:`search_workflows`
- Performance optimization: :doc:`../performance_tuning`
- Full API reference: :doc:`../api_reference`
