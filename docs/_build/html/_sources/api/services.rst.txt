Services
=========

Service classes for search, duplicate detection, and language detection.

Overview
~~~~~~~~

Services provide higher-level functionality:

- **SearchService**: All search operations (semantic, lexical, hybrid)
- **DuplicateDetector**: Find and cluster similar code
- **HybridSearcher**: Combined semantic + lexical search
- **LanguageDetector**: Detect programming language

Core Services
~~~~~~~~~~~~~

**SearchService** (``search_service.py``)

.. code-block:: python

    from gigacode.search_service import SearchService
    
    service = SearchService(
        embedder=embedder,
        index_manager=index_manager
    )
    
    # Semantic search
    results = service.semantic_search(
        buffer_id="my_project",
        query="find database operations",
        top_k=5
    )
    
    # Lexical search
    results = service.lexical_search(
        buffer_id="my_project",
        query="def connect",
        top_k=5
    )
    
    # Hybrid search
    results = service.hybrid_search(
        buffer_id="my_project",
        query="database operations",
        top_k=5,
        semantic_weight=0.7
    )

**DuplicateDetector** (``duplicate_detector.py``)

.. code-block:: python

    from gigacode.duplicate_detector import DuplicateDetector
    
    detector = DuplicateDetector()
    
    # Find duplicates
    duplicates = detector.find_duplicates(
        buffer_id="my_project",
        threshold=0.95
    )
    
    # Cluster similar code
    clusters = detector.cluster_code(
        buffer_id="my_project",
        n_clusters=10
    )

**LanguageDetector** (``language_detect.py``)

.. code-block:: python

    from gigacode.language_detect import LanguageDetector
    
    detector = LanguageDetector()
    
    # Detect language
    language = detector.detect(file_path="example.py")
    print(f"Language: {language}")  # "python"
    
    # Get language for extension
    language = detector.get_language_for_extension(".py")
    print(f"Language: {language}")  # "python"

**HybridSearcher** (``hybrid_search.py``)

.. code-block:: python

    from gigacode.hybrid_search import HybridSearcher
    
    searcher = HybridSearcher(
        semantic_search_fn=service.semantic_search,
        lexical_search_fn=service.lexical_search
    )
    
    # Combine results
    results = searcher.search(
        buffer_id="my_project",
        query="find error handling",
        semantic_weight=0.6,
        lexical_weight=0.4
    )

Phase 3 Integration
~~~~~~~~~~~~~~~~~~~

**SearchService** integrates Phase 3 optimizations:

1. **SemanticQueryCache**: Automatic query caching
2. **FAISS Optimization**: Auto-selected index types

.. code-block:: python

    service = SearchService(
        embedder=embedder,
        index_manager=index_manager,
        semantic_cache_max_entries=500,
        semantic_cache_threshold=0.95
    )
    
    # First search: computes embedding
    # Subsequent similar searches: cached (100x faster)
    results = service.semantic_search(buffer_id, query)

Configuration
~~~~~~~~~~~~~

Services are automatically configured by CodeEmbeddingTool, but you can customize:

.. code-block:: python

    from gigacode.search_service import SearchService
    
    service = SearchService(
        embedder=embedder,
        index_manager=index_manager,
        semantic_cache_max_entries=500,
        semantic_cache_threshold=0.95,
        use_gpu=True
    )

See Also
~~~~~~~~

- :doc:`search_service` - Detailed SearchService docs
- :class:`gigacode.duplicate_detector.DuplicateDetector` - Duplicate detection
- :class:`gigacode.language_detect.LanguageDetector` - Language detection
- :doc:`../tutorials/search_workflows` - Search examples
