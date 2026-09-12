FROM python:3.12-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy only what's needed for install
COPY pyproject.toml README.md LICENSE VERSION CHANGELOG.md ./
COPY gigacode/ gigacode/

# CPU-only PyTorch (avoids the multi-GB CUDA wheels), then the app with server
# extras, the CPU FAISS wheel, and the embedding stack. All dependencies ship
# manylinux wheels, so no compiler toolchain is installed.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[server]" faiss-cpu \
       "sentence-transformers>=5.0.0" "transformers>=5.8.0" "scikit-learn>=1.6.0"

# Pre-download the default embedding model (~160MB)
RUN python -c "from gigacode.embedder import Embedder; Embedder()"

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health')" || exit 1

ENTRYPOINT ["gigacode-server", "--host", "0.0.0.0", "--port", "8765"]
