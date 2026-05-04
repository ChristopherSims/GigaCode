"""OpenTelemetry metrics exporter with Prometheus HTTP endpoint (Phase 7).

Exposes collected metrics via HTTP endpoint in Prometheus format.
Supports real-time metrics scraping from monitoring systems (Prometheus, Grafana).
"""

from __future__ import annotations

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PrometheusMetricsExporter:
    """Exports metrics to Prometheus via HTTP endpoint.
    
    Features:
    - Prometheus-compatible text format output
    - Real-time metric scraping support
    - Thread-safe metric collection
    - Automatic HTTP server management
    
    Usage:
        exporter = PrometheusMetricsExporter(port=9090)
        exporter.start()
        # Server running on http://localhost:9090/metrics
        exporter.stop()
    """
    
    def __init__(self, port: int = 9090, addr: str = "0.0.0.0"):
        """Initialize Prometheus metrics exporter.
        
        Args:
            port: Port to expose /metrics endpoint (default 9090)
            addr: Address to bind HTTP server (default 0.0.0.0)
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus-client not installed. Install with: pip install prometheus-client")
        
        self.port = port
        self.addr = addr
        self.registry = CollectorRegistry()
        self._server = None
        self._server_thread = None
        self._lock = threading.Lock()
        
        # Define Prometheus metrics
        self.operations_total = Counter(
            'gigacode_operations_total',
            'Total operations performed',
            ['operation', 'status'],
            registry=self.registry,
        )
        
        self.operation_duration_seconds = Histogram(
            'gigacode_operation_duration_seconds',
            'Operation duration in seconds',
            ['operation'],
            registry=self.registry,
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
        )
        
        self.chunks_processed = Counter(
            'gigacode_chunks_processed_total',
            'Total chunks processed',
            ['operation'],
            registry=self.registry,
        )
        
        self.embedding_dimension = Gauge(
            'gigacode_embedding_dimension',
            'Dimension of embeddings',
            registry=self.registry,
        )
        
        self.cache_hits_total = Counter(
            'gigacode_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry,
        )
        
        self.cache_misses_total = Counter(
            'gigacode_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry,
        )
        
        self.cache_size_bytes = Gauge(
            'gigacode_cache_size_bytes',
            'Current cache size in bytes',
            ['cache_type'],
            registry=self.registry,
        )
        
        self.gpu_memory_usage_bytes = Gauge(
            'gigacode_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            registry=self.registry,
        )
        
        self.buffers_registered = Gauge(
            'gigacode_buffers_registered',
            'Number of registered code buffers',
            registry=self.registry,
        )
        
        self.buffers_loaded = Gauge(
            'gigacode_buffers_loaded',
            'Number of loaded buffers in memory',
            registry=self.registry,
        )
    
    def start(self) -> None:
        """Start the HTTP metrics endpoint server."""
        if self._server is not None:
            logger.warning("Metrics exporter already running on port %d", self.port)
            return
        
        with self._lock:
            handler = self._create_metrics_handler()
            self._server = HTTPServer((self.addr, self.port), handler)
            self._server_thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name="PrometheusMetricsServer",
            )
            self._server_thread.start()
            logger.info(
                "Prometheus metrics endpoint started: http://%s:%d/metrics",
                self.addr if self.addr != "0.0.0.0" else "localhost",
                self.port,
            )
    
    def stop(self) -> None:
        """Stop the HTTP metrics endpoint server."""
        with self._lock:
            if self._server is not None:
                self._server.shutdown()
                self._server = None
                self._server_thread = None
                logger.info("Prometheus metrics endpoint stopped")
    
    def _create_metrics_handler(self):
        """Create HTTP request handler for /metrics endpoint."""
        exporter = self
        registry = self.registry
        
        class MetricsHandler(BaseHTTPRequestHandler):
            """HTTP handler for Prometheus metrics endpoint."""
            
            def do_GET(self):
                """Handle GET request to /metrics."""
                if self.path == "/metrics":
                    try:
                        metrics_output = generate_latest(registry)
                        self.send_response(200)
                        self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                        self.send_header("Content-Length", len(metrics_output))
                        self.end_headers()
                        self.wfile.write(metrics_output)
                    except Exception as e:
                        logger.error("Error generating metrics: %s", e)
                        self.send_response(500)
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(f"Error: {e}".encode())
                elif self.path == "/health":
                    # Health check endpoint
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"status": "healthy"}')
                else:
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"Not Found")
            
            def log_message(self, format, *args):
                """Suppress default HTTP server logging."""
                # Log only errors, not every request
                if "code" in format and args and args[0] >= 400:
                    logger.debug(format, *args)
        
        return MetricsHandler
    
    def record_operation(
        self,
        operation: str,
        duration_s: float,
        status: str = "ok",
        chunk_count: int = 0,
    ) -> None:
        """Record an operation metric.
        
        Args:
            operation: Operation name (e.g., 'embed_codebase', 'semantic_search')
            duration_s: Operation duration in seconds
            status: Operation status ('ok', 'error', 'conflict')
            chunk_count: Number of chunks processed (optional)
        """
        self.operations_total.labels(operation=operation, status=status).inc()
        self.operation_duration_seconds.labels(operation=operation).observe(duration_s)
        
        if chunk_count > 0:
            self.chunks_processed.labels(operation=operation).inc(chunk_count)
    
    def record_cache_access(
        self,
        cache_type: str,
        hit: bool,
        size_bytes: int = 0,
    ) -> None:
        """Record a cache access metric.
        
        Args:
            cache_type: Cache type ('index', 'lexical', 'query')
            hit: Whether it was a cache hit
            size_bytes: Current cache size (optional)
        """
        if hit:
            self.cache_hits_total.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses_total.labels(cache_type=cache_type).inc()
        
        if size_bytes > 0:
            self.cache_size_bytes.labels(cache_type=cache_type).set(size_bytes)
    
    def set_embedding_dimension(self, dimension: int) -> None:
        """Set embedding dimension metric."""
        self.embedding_dimension.set(dimension)
    
    def set_gpu_memory_usage(self, bytes_used: int) -> None:
        """Set GPU memory usage metric."""
        self.gpu_memory_usage_bytes.set(bytes_used)
    
    def set_buffer_counts(self, registered: int, loaded: int) -> None:
        """Set buffer count metrics."""
        self.buffers_registered.set(registered)
        self.buffers_loaded.set(loaded)


# Global singleton exporter instance
_exporter: PrometheusMetricsExporter | None = None
_exporter_lock = threading.Lock()


def get_prometheus_exporter(port: int = 9090) -> PrometheusMetricsExporter:
    """Get or create global Prometheus metrics exporter.
    
    Args:
        port: Port for metrics endpoint (default 9090)
    
    Returns:
        PrometheusMetricsExporter instance
    """
    global _exporter
    
    if _exporter is None:
        with _exporter_lock:
            if _exporter is None:
                try:
                    _exporter = PrometheusMetricsExporter(port=port)
                    logger.info("Prometheus metrics exporter initialized (port %d)", port)
                except ImportError as e:
                    logger.warning("Prometheus not available: %s", e)
                    raise
    
    return _exporter


def configure_prometheus(port: int = 9090, start_server: bool = True) -> PrometheusMetricsExporter:
    """Configure and optionally start Prometheus metrics export.
    
    Args:
        port: Port for metrics endpoint (default 9090)
        start_server: Whether to start HTTP server immediately
    
    Returns:
        PrometheusMetricsExporter instance
    
    Example:
        exporter = configure_prometheus(port=9090, start_server=True)
        # Server now running at http://localhost:9090/metrics
    """
    exporter = get_prometheus_exporter(port=port)
    
    if start_server:
        exporter.start()
    
    return exporter
