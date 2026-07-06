FROM python:3.12-slim

WORKDIR /app

# Install build deps for faiss-cpu and tree-sitter grammars
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only what's needed for install
COPY pyproject.toml README.md LICENSE VERSION CHANGELOG.md ./
COPY gigacode/ gigacode/

# Install with embed + server extras
RUN pip install --no-cache-dir ".[embed,server]"

# Pre-download the default embedding model (~160MB)
RUN python -c "from gigacode.embedder import Embedder; Embedder()"

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health')" || exit 1

ENTRYPOINT ["gigacode-server", "--host", "0.0.0.0", "--port", "8765"]