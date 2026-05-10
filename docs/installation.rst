Installation
=============

This guide covers installing GigaCode on Windows, macOS, and Linux.

Prerequisites
~~~~~~~~~~~~~

- Python 3.10 or higher
- pip package manager
- Optional: CUDA 11.8+ (for GPU acceleration)
- Optional: Git (for development installation)

Standard Installation
~~~~~~~~~~~~~~~~~~~~~

**Step 1: Clone the repository**

.. code-block:: bash

    git clone https://github.com/your-repo/gigacode.git
    cd gigacode

**Step 2: Install GigaCode**

CPU-only (default):

.. code-block:: bash

    pip install .

With GPU support (requires CUDA):

.. code-block:: bash

    pip install ".[gpu]"

Development mode (editable install):

.. code-block:: bash

    pip install ".[dev]"

**Step 3: Verify installation**

.. code-block:: bash

    python -c "from gigacode import CodeEmbeddingTool; print('GigaCode installed')"

GPU Setup (Optional)
~~~~~~~~~~~~~~~~~~~~

**GPU Requirements:**

- NVIDIA GPU with CUDA Compute Capability 3.5+
- CUDA Toolkit 11.8 or higher
- cuDNN 8.x

**Check your GPU:**

.. code-block:: bash

    python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

**Install GPU support:**

.. code-block:: bash

    pip install ".[gpu]"
    # or with specific CUDA version
    pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118

**Verify GPU:**

.. code-block:: bash

    python -c "import torch; assert torch.cuda.is_available(), 'GPU not detected'; print(f'✓ GPU: {torch.cuda.get_device_name()}')"

Docker Installation
~~~~~~~~~~~~~~~~~~~

Build a Docker image with GigaCode:

.. code-block:: dockerfile

    FROM python:3.11
    WORKDIR /app
    COPY . .
    RUN pip install .
    RUN pip install notebook  # optional
    EXPOSE 8888
    CMD ["python", "-m", "jupyter", "notebook", "--ip=0.0.0.0"]

Build and run:

.. code-block:: bash

    docker build -t gigacode .
    docker run -it gigacode

From PyPI (Future)
~~~~~~~~~~~~~~~~~~~

Once released on PyPI:

.. code-block:: bash

    pip install gigacode

Verify Installation
~~~~~~~~~~~~~~~~~~~

**Check all dependencies:**

.. code-block:: bash

    python -c "
    from gigacode import CodeEmbeddingTool
    import torch
    import faiss
    import sentence_transformers
    print(' All dependencies OK')
    print(f'  Python: {__import__('sys').version}')
    print(f'  PyTorch: {torch.__version__}')
    print(f'  FAISS: {faiss.__version__}')
    print(f'  Sentence-Transformers: {sentence_transformers.__version__}')
    "

**Test basic functionality:**

.. code-block:: bash

    python -c "
    from gigacode import CodeEmbeddingTool
    tool = CodeEmbeddingTool(work_dir='/tmp/gigacode_test')
    print(' CodeEmbeddingTool initialized')
    print(f'  Version: {tool.__class__.__module__}')
    "

Troubleshooting
~~~~~~~~~~~~~~~

**ModuleNotFoundError: No module named 'gigacode'**

Ensure you're in the right directory and installed properly:

.. code-block:: bash

    cd /path/to/gigacode
    pip install -e .

**ImportError: No module named 'torch'**

Install PyTorch:

.. code-block:: bash

    pip install torch==2.0.0

**CUDA not available (GPU mode)**

Check CUDA installation:

.. code-block:: bash

    # Windows
    cd C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
    ./nvidia-smi.exe
    
    # Linux/macOS
    nvidia-smi

If nvidia-smi fails, install CUDA Toolkit from https://developer.nvidia.com/cuda-toolkit

**Out of memory**

- Use CPU mode: ``pip install .`` (skip GPU)
- Reduce buffer size: ``CodeEmbeddingTool(max_buffers=2)``
- Process fewer files per batch

**Slow search performance**

First search is slower (index loading). Subsequent searches are cached and faster. Enable GPU for 2-10x speedup:

.. code-block:: bash

    pip install ".[gpu]"

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development, install with test dependencies:

.. code-block:: bash

    pip install ".[dev]"

Run tests:

.. code-block:: bash

    pytest tests/ -v

Build documentation:

.. code-block:: bash

    cd docs
    make html

System-Specific Notes
~~~~~~~~~~~~~~~~~~~~~

**Windows:**

- Use PowerShell or Command Prompt
- Paths use backslashes (automatic in Python)
- For GPU: Ensure Visual C++ Redistributable is installed

**macOS:**

- Works on Intel and Apple Silicon (M1/M2/M3)
- GPU support requires external NVIDIA GPU via Thunderbolt
- Most development is CPU-only

**Linux:**

- Full GPU support for NVIDIA GPUs
- On some systems, you may need: ``apt install python3-dev``
- Docker recommended for reproducible environments

Next Steps
~~~~~~~~~~

1. Read :doc:`quick_start` for your first search
2. Check out :doc:`tutorials/basic_embed` for detailed examples
3. Explore :doc:`api_reference` for the complete API

Questions?
~~~~~~~~~~

- Check :doc:`performance_tuning` for optimization strategies
- Search documentation at :ref:`search`
- Open an issue on GitHub
