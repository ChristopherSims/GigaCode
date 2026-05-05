FAISS Optimizer
================

``gigacode.faiss_optimizer.FAISSIndexOptimizer`` automatically selects optimal FAISS index types for smart performance tuning.

Overview
~~~~~~~~

The FAISS Optimizer is responsible for:

- Auto-selecting optimal index type based on vector count
- Parameterizing index configuration
- Benchmarking different index types
- Providing index statistics and info

Key Benefits
~~~~~~~~~~~~

- **Smart selection**: Index type chosen based on data size
- **Automatic tuning**: Parameters optimized for your data
- **Future flexibility**: Easily switch index types
- **Performance predictions**: Know search latency before running

Index Types
~~~~~~~~~~~

GigaCode supports three FAISS index types, auto-selected based on vector count:

**Flat (<10k vectors)**

- Type: Exact nearest neighbor search
- Memory: Low
- Speed: Fast for small indexes
- Use case: Small projects, development

.. code-block:: python

    index = optimizer.create_optimized_index(
        dim=384,
        vector_count=5000  # Uses Flat
    )

**IVF (10k-100k vectors)**

- Type: Inverted file (partitioned search)
- Memory: Medium
- Speed: Fast for medium indexes
- Use case: Most projects
- Parameters: nlist = sqrt(N), nprobe = nlist/10

.. code-block:: python

    index = optimizer.create_optimized_index(
        dim=384,
        vector_count=50000  # Uses IVF
    )

**HNSW (>100k vectors)**

- Type: Hierarchical navigable small world
- Memory: Medium-High
- Speed: Very fast for large indexes
- Use case: Large projects
- Parameters: nlinks = 16, efConstruction = 200

.. code-block:: python

    index = optimizer.create_optimized_index(
        dim=384,
        vector_count=500000  # Uses HNSW
    )

Core Class
~~~~~~~~~~

.. autoclass:: gigacode.faiss_optimizer.FAISSIndexOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

Auto-Selection Logic
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gigacode.faiss_optimizer import FAISSIndexOptimizer
    
    optimizer = FAISSIndexOptimizer()
    
    # Auto-select index type
    index_type = optimizer.select_index_type(vector_count=50000)
    print(f"Selected: {index_type}")  # "IVF"
    
    # Get parameters for selected type
    params = optimizer.get_index_params(vector_count=50000)
    print(f"Parameters: {params}")
    # {
    #     "index_type": "IVF",
    #     "nlist": 223,  # sqrt(50000)
    #     "nprobe": 22   # sqrt(50000)/10
    # }

Create Optimized Index
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gigacode.faiss_optimizer import FAISSIndexOptimizer
    
    optimizer = FAISSIndexOptimizer()
    
    # Create index for your data
    index = optimizer.create_optimized_index(
        dim=384,                    # Embedding dimension
        vector_count=50000,         # Number of vectors
        use_gpu=True                # GPU acceleration
    )
    
    # Add vectors
    import numpy as np
    vectors = np.random.randn(50000, 384).astype('float32')
    index.add(vectors)
    
    # Search
    query = np.random.randn(1, 384).astype('float32')
    distances, indices = index.search(query, k=5)

Manual Index Selection
~~~~~~~~~~~~~~~~~~~~~~

You can manually override auto-selection:

.. code-block:: python

    from gigacode.gpu_index import GpuIndex
    
    # Force specific index type
    index = GpuIndex(
        dim=384,
        index_type="ivf"  # Force IVF even if not auto-selected
    )

Index Information
~~~~~~~~~~~~~~~~~

Get information about an index:

.. code-block:: python

    info = optimizer.get_index_info(index)
    print(f"Type: {info['index_type']}")
    print(f"Vectors: {info['ntotal']}")
    print(f"Dimension: {info['dim']}")
    print(f"GPU: {info['use_gpu']}")

Search with Index
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    
    # Search for K nearest neighbors
    query = np.array([[...]], dtype='float32')  # 1x384
    
    # Different methods by index type
    distances, indices = index.search(query, k=5)
    
    for idx, distance in zip(indices[0], distances[0]):
        print(f"Match {idx}: distance {distance:.3f}")

Configuration
~~~~~~~~~~~~~

.. code-block:: python

    from gigacode.faiss_optimizer import FAISSIndexOptimizer
    
    optimizer = FAISSIndexOptimizer(
        use_gpu=True,              # GPU acceleration
        flat_threshold=10000,      # Threshold for Flat
        ivf_threshold=100000,      # Threshold for IVF
    )

Performance Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Flat Index:**

- Build time: O(N)
- Search time: O(N*d)
- Memory: 4*N*d bytes
- 5k vectors: <1ms search
- 10k vectors: 1-2ms search

**IVF Index:**

- Build time: O(N*log(N))
- Search time: O(nlist*nprobe*d)
- Memory: ~4*N*d bytes
- 50k vectors: 5-10ms search
- 100k vectors: 10-20ms search

**HNSW Index:**

- Build time: O(N*log(N))
- Search time: O(log(N)) + constant
- Memory: 4*N*d + overhead
- 500k vectors: 2-5ms search (faster than IVF!)
- 1M+ vectors: 3-10ms search

Tuning Thresholds
~~~~~~~~~~~~~~~~~

Adjust when different index types are selected:

.. code-block:: python

    optimizer = FAISSIndexOptimizer(
        flat_threshold=5000,      # Lower = Flat for smaller projects
        ivf_threshold=50000       # Lower = IVF for smaller projects
    )
    
    # Now:
    # <5k vectors -> Flat
    # 5k-50k vectors -> IVF
    # >50k vectors -> HNSW

Example: Index Selection at Different Scales
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gigacode.faiss_optimizer import FAISSIndexOptimizer
    
    optimizer = FAISSIndexOptimizer()
    
    test_sizes = [1000, 5000, 10000, 50000, 100000, 500000]
    
    for size in test_sizes:
        index_type = optimizer.select_index_type(size)
        params = optimizer.get_index_params(size)
        print(f"{size:>7} vectors -> {index_type:>4} "
              f"(params: {params})")
    
    # Output:
    #    1000 vectors -> flat (no params)
    #    5000 vectors -> flat (no params)
    #   10000 vectors -> ivf (nlist: 100, nprobe: 10)
    #   50000 vectors -> ivf (nlist: 223, nprobe: 22)
    #  100000 vectors -> ivf (nlist: 316, nprobe: 31)
    #  500000 vectors -> hnsw (nlinks: 16, efConstruction: 200)

Benchmarking
~~~~~~~~~~~~

Compare index types:

.. code-block:: python

    import numpy as np
    import time
    
    # Create test data
    dim = 384
    vectors = np.random.randn(50000, dim).astype('float32')
    queries = np.random.randn(100, dim).astype('float32')
    
    optimizer = FAISSIndexOptimizer()
    
    for index_type in ["flat", "ivf", "hnsw"]:
        # Create index
        if index_type == "flat":
            index = optimizer.create_optimized_index(dim, len(vectors), index_type="flat")
        # ... other types
        
        index.add(vectors)
        
        # Benchmark search
        start = time.time()
        for query in queries:
            index.search(query.reshape(1, -1), k=5)
        elapsed = time.time() - start
        
        avg_per_query = elapsed / len(queries)
        print(f"{index_type}: {avg_per_query*1000:.2f}ms per query")

Integration with GpuIndex
~~~~~~~~~~~~~~~~~~~~~~~~~

GpuIndex automatically uses FAISSIndexOptimizer:

.. code-block:: python

    from gigacode.gpu_index import GpuIndex
    
    # Auto-selection based on vector count
    gpu_index = GpuIndex(
        dim=384,
        embeddings=embeddings  # Vector count determines type
    )
    
    # Manual override
    gpu_index = GpuIndex(
        dim=384,
        embeddings=embeddings,
        index_type="ivf"  # Force IVF
    )

Future Enhancements
~~~~~~~~~~~~~~~~~~~

The infrastructure is in place for future optimizations:

- Adaptive index type switching as data grows
- Automatic parameter tuning based on workload
- Mixed index types for heterogeneous data
- Quantization for memory reduction

Troubleshooting
~~~~~~~~~~~~~~~

**Index type seems wrong:**

- Check vector count: too small uses Flat, too large uses HNSW
- Verify thresholds: adjust flat_threshold and ivf_threshold

**Search is slow:**

- Check index type with ``get_index_info()``
- May need to switch to GPU: ``use_gpu=True``
- Try different nlist/nprobe for IVF

**Out of memory:**

- Try Flat (smallest memory)
- Reduce vector dimension
- Use quantization (future feature)

See Also
~~~~~~~~

- :class:`gigacode.gpu_index.GpuIndex` - GPU index implementation
- :class:`gigacode.incremental_indexer.IncrementalIndexManager` - Incremental updates
- :doc:`../performance_tuning` - Performance optimization guide
