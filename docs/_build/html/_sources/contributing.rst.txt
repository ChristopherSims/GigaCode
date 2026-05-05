Contributing
=============

Guidelines for contributing to GigaCode.

Getting Started
~~~~~~~~~~~~~~~

**1. Fork and clone:**

.. code-block:: bash

    git clone https://github.com/your-fork/gigacode.git
    cd gigacode
    git remote add upstream https://github.com/original/gigacode.git

**2. Create virtual environment:**

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

**3. Install development dependencies:**

.. code-block:: bash

    pip install ".[dev]"

**4. Run tests to verify setup:**

.. code-block:: bash

    pytest tests/ -v

Development Workflow
~~~~~~~~~~~~~~~~~~~~

**Creating a feature branch:**

.. code-block:: bash

    git checkout -b feature/my-feature
    # Make your changes
    git add .
    git commit -m "feat: description of changes"

**Naming conventions:**

- Branches: ``feature/description`` or ``fix/description``
- Commits: ``feat:`` (feature), ``fix:`` (bugfix), ``docs:`` (documentation)
- PRs: Clear, descriptive title

**Commit messages:**

.. code-block:: text

    feat: add new search filter option
    
    - Allows filtering results by file extension
    - Improves search performance for specific file types
    - Adds 5 new tests for filter functionality

Code Style
~~~~~~~~~~

**Follow PEP 8:**

.. code-block:: bash

    # Format code
    black gigacode/ tests/
    
    # Check style
    flake8 gigacode/ tests/

**Type hints:**

.. code-block:: python

    # Good: clear types
    def search(self, query: str, top_k: int = 5) -> List[Match]:
        pass
    
    # Bad: missing types
    def search(self, query, top_k=5):
        pass

**Docstrings:**

.. code-block:: python

    def semantic_search(
        self,
        buffer_id: str,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Search semantically for code matching query.
        
        Args:
            buffer_id: Buffer identifier
            query: Search query in natural language
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
        
        Returns:
            Dictionary with 'matches' and 'total' keys
            
        Raises:
            BufferNotFoundError: If buffer_id doesn't exist
            
        Example:
            >>> results = tool.semantic_search(
            ...     buffer_id="my_project",
            ...     query="find database connections",
            ...     top_k=5
            ... )
            >>> print(f"Found {results['total']} matches")
        """

Testing
~~~~~~~

**Write tests for new features:**

.. code-block:: python

    import pytest
    from gigacode import CodeEmbeddingTool
    
    def test_new_feature():
        """Test description."""
        tool = CodeEmbeddingTool()
        
        # Arrange
        buffer_id = tool.embed_codebase("/test/project")
        
        # Act
        result = tool.new_feature(buffer_id)
        
        # Assert
        assert result is not None
        assert result["success"] == True

**Run tests:**

.. code-block:: bash

    # All tests
    pytest tests/
    
    # Specific test
    pytest tests/test_feature.py::test_function
    
    # With coverage
    pytest --cov=gigacode tests/

**Test organization:**

- Unit tests: Single component behavior
- Integration tests: Multi-component interaction
- Performance tests: Benchmarks

Documentation
~~~~~~~~~~~~~~

**Update documentation when:**

- Adding new features
- Changing APIs
- Fixing bugs that affect usage

**Build documentation locally:**

.. code-block:: bash

    cd docs
    pip install sphinx sphinx-rtd-theme
    make html
    open _build/html/index.html

**Documentation should include:**

- API reference (auto-generated from docstrings)
- Usage examples
- Configuration options
- Performance notes

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Optimizations to consider:**

- Use GPU when embedding large data
- Cache expensive computations
- Batch operations when possible
- Avoid unnecessary file I/O

**Performance testing:**

.. code-block:: python

    import time
    
    start = time.time()
    # Code to benchmark
    elapsed = time.time() - start
    
    # Assert performance is acceptable
    assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s"

Phase 3 Optimizations Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When adding features, consider:

**1. Incremental Indexing**

If your feature involves index updates, it automatically benefits from incremental indexing. Just ensure you're using IndexManager correctly:

.. code-block:: python

    # Automatic incremental update
    manager._rebuild_files()  # Uses IncrementalIndexManager if available

**2. Semantic Caching**

If your feature involves queries, it automatically benefits from semantic caching:

.. code-block:: python

    # Automatic query caching
    results = service.semantic_search(query)  # Uses SemanticQueryCache

**3. FAISS Optimization**

When creating indices, they automatically use optimal settings:

.. code-block:: python

    # Automatic index type selection
    index = optimizer.create_optimized_index(dim=384, vector_count=50000)

Code Review Process
~~~~~~~~~~~~~~~~~~~

**Before submitting PR:**

1. Run all tests: ``pytest tests/ -v``
2. Check code style: ``black --check gigacode/``
3. Update documentation
4. Add tests for new functionality
5. Run performance benchmarks

**PR review checklist:**

- [ ] Tests pass
- [ ] Code style compliant
- [ ] Documentation updated
- [ ] Performance acceptable
- [ ] Backward compatible

**Review guidelines:**

- Be respectful and constructive
- Focus on code, not person
- Ask clarifying questions
- Suggest improvements
- Approve when satisfied

Common Contributing Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Adding a new feature:**

1. Create branch: ``git checkout -b feature/my-feature``
2. Implement feature with tests
3. Update docstrings and docs
4. Run full test suite
5. Submit PR with description

**Fixing a bug:**

1. Create branch: ``git checkout -b fix/bug-description``
2. Write test that reproduces bug
3. Fix the bug
4. Verify test passes
5. Submit PR with issue reference: ``Fixes #123``

**Improving performance:**

1. Benchmark current performance
2. Implement optimization
3. Benchmark improvement
4. Add performance regression tests
5. Document in performance_tuning.rst

**Adding documentation:**

1. Create or update .rst file
2. Build docs: ``cd docs && make html``
3. Verify rendering: ``open _build/html/index.html``
4. Submit PR

Reporting Issues
~~~~~~~~~~~~~~~~

**Before opening issue, check:**

- Not a duplicate
- Using latest version
- Have minimal reproduction case

**Good issue report includes:**

.. code-block:: text

    Title: Brief description
    
    Environment:
    - OS: Windows 10
    - Python: 3.11
    - CUDA: 11.8
    
    Steps to reproduce:
    1. ...
    2. ...
    3. ...
    
    Expected: ...
    Actual: ...
    
    Minimal code example:
    ```python
    # Code here
    ```

Release Process
~~~~~~~~~~~~~~~

Releases follow semantic versioning (major.minor.patch):

- Major: Breaking changes
- Minor: New features, backward compatible
- Patch: Bug fixes

**Release checklist:**

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] Tag created: ``git tag v1.0.0``
- [ ] Released to PyPI

Recognition
~~~~~~~~~~~

All contributors are recognized in:

- CONTRIBUTORS.md file
- Release notes
- GitHub contributors page

Large contributions may warrant:

- Co-authorship on relevant commits
- Recognition in documentation
- Special mention in announcements

Need Help?
~~~~~~~~~~

- **Questions?** Open a discussion on GitHub
- **Bug?** Create an issue with reproduction
- **Feature idea?** Open a discussion first
- **Not sure?** Ask in issue comments

Community
~~~~~~~~~

- Be respectful to all contributors
- Help others with questions
- Share knowledge and experience
- Give constructive feedback

Code of Conduct
~~~~~~~~~~~~~~~

All contributors agree to:

- Treat others with respect
- Welcome newcomers
- Give credit where due
- Provide supportive feedback
- Report violations to maintainers

Thank you for contributing!
~~~~~~~~~~~~~~~~~~~~~~~~~

Your contributions make GigaCode better for everyone. Whether it's code, documentation, bug reports, or suggestions - we appreciate your help!

Next Steps
~~~~~~~~~~

- Read :doc:`architecture` for design overview
- Check :doc:`performance_tuning` for optimization patterns
- Review existing tests in ``tests/`` for examples
- Open a discussion before major changes

Questions?
~~~~~~~~~~

Feel free to:

- Open an issue on GitHub
- Start a discussion
- Comment on existing issues
- Ask in pull request comments
