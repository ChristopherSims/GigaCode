Services
=========

Service classes and helpers for search, duplicate detection, and language
detection.

Overview
~~~~~~~~

- :class:`gigacode.search_service.SearchService` — all search operations
  (semantic, lexical, hybrid).
- :func:`gigacode.duplicate_detector.find_duplicates` — MinHash/LSH near-duplicate detection.
- :func:`gigacode.hybrid_search.reciprocal_rank_fusion` — combine ranked result lists.
- :func:`gigacode.language_detect.detect_language` — detect a file's language.

SearchService
~~~~~~~~~~~~~

.. code-block:: python

    from gigacode.search_service import SearchService

    service = SearchService(
        embedder=embedder,
        index_manager=index_manager,
    )

    results = service.semantic_search(
        buffer_id="my_project",
        query="find database operations",
        top_k=5,
    )

    results = service.hybrid_search(
        buffer_id="my_project",
        query="database operations",
        top_k=5,
        semantic_weight=0.7,
    )

Duplicate detection
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gigacode.duplicate_detector import find_duplicates

    duplicates = find_duplicates(chunks, threshold=0.9)

Hybrid ranking
~~~~~~~~~~~~~~

.. code-block:: python

    from gigacode.hybrid_search import reciprocal_rank_fusion

    merged = reciprocal_rank_fusion(
        semantic_results=semantic,
        lexical_results=lexical,
        semantic_weight=0.6,
        lexical_weight=0.4,
        top_k=10,
    )

Language detection
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gigacode.language_detect import detect_language

    language = detect_language("example.py")
    print(f"Language: {language}")  # "python"

See Also
~~~~~~~~

- :doc:`search_service` - Detailed SearchService docs
- :doc:`duplicate_detector` - Duplicate detection
- :doc:`language_detect` - Language detection
- :doc:`../tutorials/search_workflows` - Search examples
