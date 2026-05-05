Search Workflows
================

This tutorial covers different ways to search embedded code, from basic semantic search to advanced hybrid techniques.

Overview
~~~~~~~~

GigaCode supports multiple search methods:

1. **Semantic Search** - Find code by meaning and intent
2. **Lexical Search** - Find code by exact keywords
3. **Hybrid Search** - Combine semantic and lexical for best results
4. **Filtering** - Narrow results by language, file, or other criteria

Prerequisites
~~~~~~~~~~~~~

Make sure you have embedded a project. If not, see :doc:`basic_embed`:

.. code-block:: python

    from gigacode import CodeEmbeddingTool
    
    tool = CodeEmbeddingTool()
    buffer_id = tool.embed_codebase("/path/to/project")

Basic Semantic Search
~~~~~~~~~~~~~~~~~~~~~

Search by meaning, not keywords:

.. code-block:: python

    # Find code related to a concept
    results = tool.semantic_search(
        buffer_id="my_project",
        query="find the function that initializes the database",
        top_k=5
    )
    
    # Display results
    for i, match in enumerate(results["matches"], 1):
        print(f"\n{i}. {match['name']} ({match['file']}:{match['start_line']})")
        print(f"   Score: {match['score']:.2f}")
        print(f"   {match['text'][:100]}...")

**Key parameters:**

- ``query`` (str): What you're looking for (in natural language)
- ``top_k`` (int): Number of results (default: 5)
- ``threshold`` (float): Minimum similarity score (0-1, default: 0.0)

**How semantic search works:**

1. Your query is converted to a vector embedding
2. Similarity to all code chunks is calculated
3. Top matches are returned, sorted by similarity

**Semantic vs. lexical:**

.. code-block:: python

    # Semantic: finds by meaning
    tool.semantic_search(
        buffer_id="my_project",
        query="send HTTP request to remote server"
    )
    # Returns: requests.get(), urllib.urlopen(), httplib calls, etc.
    
    # Lexical: finds by keywords only
    tool.lexical_search(
        buffer_id="my_project",
        query="HTTP request"
    )
    # Returns: only code containing exact words "HTTP" and "request"

Advanced Search Options
~~~~~~~~~~~~~~~~~~~~~~~~

**Filter by file type:**

.. code-block:: python

    results = tool.semantic_search(
        buffer_id="my_project",
        query="error handling",
        top_k=10,
        file_patterns=["*.py"]  # Python files only
    )

**Filter by directory:**

.. code-block:: python

    results = tool.semantic_search(
        buffer_id="my_project",
        query="user authentication",
        top_k=5,
        include_dirs=["src/auth", "src/security"]
    )

**Filter by language:**

.. code-block:: python

    results = tool.semantic_search(
        buffer_id="my_project",
        query="async processing",
        top_k=5,
        languages=["python", "javascript"]  # Python and JS only
    )

**Set similarity threshold:**

.. code-block:: python

    # Only return very similar results
    results = tool.semantic_search(
        buffer_id="my_project",
        query="cache management",
        top_k=10,
        threshold=0.85  # 0-1, higher = more similar
    )

**Combine filters:**

.. code-block:: python

    results = tool.semantic_search(
        buffer_id="my_project",
        query="database connection pooling",
        top_k=5,
        file_patterns=["*.py"],
        exclude_dirs=["test", "examples"],
        threshold=0.8
    )

Lexical Search
~~~~~~~~~~~~~~

Find code by exact keywords:

.. code-block:: python

    # Find all code containing specific terms
    results = tool.lexical_search(
        buffer_id="my_project",
        query="redis cache",
        top_k=10
    )

**When to use lexical search:**

- Searching for specific function names
- Finding imports of a library
- Locating configuration values
- Searching for TODO or FIXME comments

**Case-sensitive matching:**

.. code-block:: python

    results = tool.lexical_search(
        buffer_id="my_project",
        query="MyClass",
        case_sensitive=True
    )

**Regex patterns:**

.. code-block:: python

    import re
    
    # Find all variable assignments
    results = tool.lexical_search(
        buffer_id="my_project",
        query=r"^\s*[a-z_]\w*\s*=",  # regex pattern
        use_regex=True
    )

Hybrid Search
~~~~~~~~~~~~~

Combine semantic and lexical for best results:

.. code-block:: python

    # Hybrid: semantic + lexical
    results = tool.hybrid_search(
        buffer_id="my_project",
        query="async function that handles network requests",
        top_k=10,
        semantic_weight=0.7,  # 70% semantic, 30% lexical
        lexical_weight=0.3
    )

**How hybrid search works:**

1. Run semantic search (finds by meaning)
2. Run lexical search (finds by keywords)
3. Combine results weighted by parameters
4. Return top results by combined score

**Tuning weights:**

.. code-block:: python

    # More semantic (good for conceptual queries)
    results = tool.hybrid_search(
        buffer_id="my_project",
        query="implement error recovery",
        semantic_weight=0.8,
        lexical_weight=0.2
    )
    
    # More lexical (good for specific code patterns)
    results = tool.hybrid_search(
        buffer_id="my_project",
        query="def __init__",
        semantic_weight=0.3,
        lexical_weight=0.7
    )

Searching with Caching
~~~~~~~~~~~~~~~~~~~~~~

Repeated searches are automatically cached:

.. code-block:: python

    # First search: slow (computes embeddings)
    results1 = tool.semantic_search(
        buffer_id="my_project",
        query="find the authentication module"
    )
    
    # Identical query: fast (from cache)
    results2 = tool.semantic_search(
        buffer_id="my_project",
        query="find the authentication module"
    )
    # This is ~100x faster!
    
    # Similar query: medium (semantic cache hit)
    results3 = tool.semantic_search(
        buffer_id="my_project",
        query="locate authentication code"  # Slightly different
    )
    # Semantic similarity detected, 50x faster

**View cache statistics:**

.. code-block:: python

    stats = tool.get_cache_stats(buffer_id="my_project")
    print(f"Cache hit rate: {stats['hit_rate']:.1%}")
    print(f"Cache entries: {stats['entries']}")
    print(f"Total searches: {stats['total_searches']}")

**Clear cache:**

.. code-block:: python

    tool.clear_cache(buffer_id="my_project")

Processing Search Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Access result fields:**

.. code-block:: python

    results = tool.semantic_search(
        buffer_id="my_project",
        query="parse JSON data",
        top_k=3
    )
    
    for match in results["matches"]:
        print(f"File: {match['file']}")
        print(f"Line: {match['start_line']}-{match['end_line']}")
        print(f"Name: {match['name']}")
        print(f"Type: {match['type']}")  # function, class, etc.
        print(f"Score: {match['score']:.3f}")
        print(f"Code:\n{match['text']}\n")

**Result structure:**

.. code-block:: python

    {
        "matches": [
            {
                "file": "src/parser.py",
                "start_line": 42,
                "end_line": 58,
                "name": "parse_json",
                "type": "function",
                "score": 0.92,
                "text": "def parse_json(data: str) -> dict:\n    # ...",
                "chunk_id": "chunk_123"
            }
        ],
        "total": 1,
        "took_ms": 12.3
    }

**Sort results:**

.. code-block:: python

    results = tool.semantic_search(
        buffer_id="my_project",
        query="user authentication",
        top_k=20
    )
    
    # Sort by similarity score
    sorted_matches = sorted(
        results["matches"],
        key=lambda m: m["score"],
        reverse=True
    )
    
    # Sort by file location
    sorted_by_file = sorted(
        results["matches"],
        key=lambda m: (m["file"], m["start_line"])
    )
    
    # Sort by code type
    sorted_by_type = sorted(
        results["matches"],
        key=lambda m: m["type"]
    )

Advanced: Clustering Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Group similar search results:

.. code-block:: python

    # Semantic search with clustering
    results = tool.semantic_search(
        buffer_id="my_project",
        query="database access",
        top_k=20
    )
    
    # Cluster the results
    clusters = tool.cluster_results(
        results["matches"],
        n_clusters=3
    )
    
    # Display clusters
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i+1} ({len(cluster)} matches):")
        for match in cluster:
            print(f"  - {match['name']} ({match['file']})")

Advanced: Finding Duplicates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Find duplicate or nearly-duplicate code:

.. code-block:: python

    # Find similar code within buffer
    duplicates = tool.find_duplicates(
        buffer_id="my_project",
        threshold=0.95,  # 0-1, higher = more similar
        top_k=5
    )
    
    for group in duplicates:
        print(f"Found {len(group)} similar chunks:")
        for chunk in group:
            print(f"  {chunk['file']}:{chunk['start_line']}")

Performance Tips
~~~~~~~~~~~~~~~~

**Speed up searches:**

1. **Use filters:** Narrow search space by file, directory, or language
2. **Reduce top_k:** Only ask for results you need
3. **Set threshold:** Only return sufficiently similar results
4. **Use GPU:** Automatic if available
5. **Semantic cache:** Repeated queries are fast

**Example: Fast search on large codebase**

.. code-block:: python

    # Narrow search to specific directory and language
    results = tool.semantic_search(
        buffer_id="huge_project",
        query="transaction handling",
        top_k=3,  # Only want 3 results
        threshold=0.85,  # Skip low-scoring matches
        include_dirs=["src/database"],
        file_patterns=["*.py"],
        languages=["python"]
    )
    # Much faster than searching entire codebase!

Complete Workflow Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a complete search workflow:

.. code-block:: python

    from gigacode import CodeEmbeddingTool
    
    # Initialize
    tool = CodeEmbeddingTool()
    buffer_id = tool.embed_codebase("/path/to/project")
    
    # Step 1: Broad semantic search
    results = tool.semantic_search(
        buffer_id=buffer_id,
        query="handle authentication errors",
        top_k=10
    )
    print(f"Found {results['total']} matches")
    
    # Step 2: Filter by score
    high_confidence = [
        m for m in results["matches"]
        if m["score"] > 0.85
    ]
    print(f"High confidence: {len(high_confidence)}")
    
    # Step 3: Narrow to specific files
    in_src = [
        m for m in high_confidence
        if m["file"].startswith("src/")
    ]
    print(f"In src/: {len(in_src)}")
    
    # Step 4: Display top result
    if in_src:
        top = in_src[0]
        print(f"\nTop match: {top['name']}")
        print(f"Location: {top['file']}:{top['start_line']}")
        print(f"Score: {top['score']:.2f}")
        print(f"\n{top['text']}")

Common Search Patterns
~~~~~~~~~~~~~~~~~~~~~~

**Find all occurrences of a pattern:**

.. code-block:: python

    results = tool.lexical_search(
        buffer_id="my_project",
        query="TODO",  # Find all TODOs
        top_k=100
    )

**Find related functionality:**

.. code-block:: python

    results = tool.semantic_search(
        buffer_id="my_project",
        query="file I/O operations",
        top_k=10
    )

**Find by function signature:**

.. code-block:: python

    results = tool.lexical_search(
        buffer_id="my_project",
        query="def.*async.*requests",
        use_regex=True,
        top_k=20
    )

**Find configuration:**

.. code-block:: python

    results = tool.lexical_search(
        buffer_id="my_project",
        query="CONFIG|config|settings",
        use_regex=True,
        top_k=10
    )

Troubleshooting
~~~~~~~~~~~~~~~

**Search returns no results:**

- Query might be too specific
- Try semantic search instead of lexical
- Lower the ``threshold`` parameter
- Remove file/directory filters
- Ensure buffer is properly embedded

**Search is slow:**

- First search loads index (normal)
- Repeated searches are cached and fast
- Use filters to narrow search space
- Check GPU is enabled for large buffers

**Results don't seem relevant:**

- Try hybrid search combining semantic + lexical
- Adjust semantic_weight and lexical_weight
- Different query wording may work better
- Use threshold to filter low-confidence results

**Too many results:**

- Increase ``threshold`` to filter low-scoring matches
- Reduce ``top_k`` parameter
- Add file/directory filters
- Use hybrid search with better weighting

Next Steps
~~~~~~~~~~

- Learn about editing and committing: :doc:`edit_workflows`
- Explore advanced indexing: :doc:`../api_reference`
- Check performance tips: :doc:`../performance_tuning`
