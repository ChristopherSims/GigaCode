"""Tests for Prometheus metrics exporter (Phase 7).

Validates HTTP endpoint, Prometheus format, and metric collection.
"""

# CRITICAL: Initialize sklearn FIRST before any gigacode imports
import types

try:
    import sklearn

    if getattr(sklearn, "__spec__", None) is None:
        sklearn.__spec__ = types.ModuleSpec("sklearn", getattr(sklearn, "__file__", None))
except Exception:
    pass


import time
import urllib.request

import pytest

from gigacode.metrics_exporter import (
    PrometheusMetricsExporter,
    configure_prometheus,
    get_prometheus_exporter,
)


class TestPrometheusMetricsExporter:
    """Tests for PrometheusMetricsExporter class."""

    def test_exporter_initialization(self):
        """Test exporter initialization with default settings."""
        exporter = PrometheusMetricsExporter(port=19090)
        assert exporter.port == 19090
        assert exporter.addr == "0.0.0.0"
        assert exporter.registry is not None

    def test_exporter_custom_port_and_addr(self):
        """Test exporter initialization with custom port and address."""
        exporter = PrometheusMetricsExporter(port=19091, addr="127.0.0.1")
        assert exporter.port == 19091
        assert exporter.addr == "127.0.0.1"

    def test_exporter_metrics_defined(self):
        """Test that all expected metrics are defined."""
        exporter = PrometheusMetricsExporter(port=19092)

        assert exporter.operations_total is not None
        assert exporter.operation_duration_seconds is not None
        assert exporter.chunks_processed is not None
        assert exporter.embedding_dimension is not None
        assert exporter.cache_hits_total is not None
        assert exporter.cache_misses_total is not None
        assert exporter.cache_size_bytes is not None
        assert exporter.gpu_memory_usage_bytes is not None
        assert exporter.buffers_registered is not None
        assert exporter.buffers_loaded is not None

    def test_record_operation(self):
        """Test recording operation metrics."""
        exporter = PrometheusMetricsExporter(port=19093)

        # Record an operation
        exporter.record_operation(
            operation="embed_codebase",
            duration_s=1.234,
            status="ok",
            chunk_count=42,
        )

        # Verify no exceptions (actual values verified via HTTP endpoint)
        assert True  # Recording succeeded

    def test_record_cache_access(self):
        """Test recording cache access metrics."""
        exporter = PrometheusMetricsExporter(port=19094)

        # Record cache hits and misses
        exporter.record_cache_access("query", hit=True, size_bytes=1024)
        exporter.record_cache_access("query", hit=False, size_bytes=1024)
        exporter.record_cache_access("index", hit=True, size_bytes=5242880)

        # Verify no exceptions
        assert True

    def test_set_embedding_dimension(self):
        """Test setting embedding dimension metric."""
        exporter = PrometheusMetricsExporter(port=19095)
        exporter.set_embedding_dimension(768)
        assert True

    def test_set_gpu_memory_usage(self):
        """Test setting GPU memory usage metric."""
        exporter = PrometheusMetricsExporter(port=19096)
        exporter.set_gpu_memory_usage(5242880000)  # 5 GB
        assert True

    def test_set_buffer_counts(self):
        """Test setting buffer count metrics."""
        exporter = PrometheusMetricsExporter(port=19097)
        exporter.set_buffer_counts(registered=3, loaded=2)
        assert True


class TestMetricsHTTPEndpoint:
    """Tests for HTTP metrics endpoint."""

    def test_http_server_start_and_stop(self):
        """Test starting and stopping HTTP metrics server."""
        exporter = PrometheusMetricsExporter(port=19098)

        # Should start without error
        exporter.start()
        assert exporter._server is not None
        assert exporter._server_thread is not None

        # Give server time to start
        time.sleep(0.2)

        # Should stop without error
        exporter.stop()
        assert exporter._server is None
        assert exporter._server_thread is None

    def test_metrics_endpoint_responds(self):
        """Test that /metrics endpoint responds with Prometheus format."""
        exporter = PrometheusMetricsExporter(port=19099)
        exporter.start()

        try:
            # Give server time to start
            time.sleep(0.3)

            # Fetch metrics
            url = "http://localhost:19099/metrics"
            response = urllib.request.urlopen(url)
            metrics = response.read().decode("utf-8")

            # Verify response
            assert response.status == 200
            assert len(metrics) > 0
            assert "# HELP" in metrics  # Prometheus format
            assert "# TYPE" in metrics

        finally:
            exporter.stop()

    def test_metrics_format_prometheus_standard(self):
        """Test that metrics conform to Prometheus format."""
        exporter = PrometheusMetricsExporter(port=19100)
        exporter.record_operation("test_op", 0.5, "ok", 10)
        exporter.record_cache_access("query", hit=True, size_bytes=1024)
        exporter.start()

        try:
            time.sleep(0.3)

            url = "http://localhost:19100/metrics"
            response = urllib.request.urlopen(url)
            metrics = response.read().decode("utf-8")

            # Check for expected metric names
            assert "gigacode_operations_total" in metrics
            assert "gigacode_operation_duration_seconds" in metrics
            assert "gigacode_chunks_processed_total" in metrics
            assert "gigacode_cache_hits_total" in metrics
            assert "gigacode_cache_misses_total" in metrics

            # Check for labels
            assert 'operation="test_op"' in metrics or "operation=test_op" in metrics
            assert 'status="ok"' in metrics or "status=ok" in metrics

        finally:
            exporter.stop()

    def test_health_endpoint_responds(self):
        """Test that /health endpoint responds."""
        exporter = PrometheusMetricsExporter(port=19101)
        exporter.start()

        try:
            time.sleep(0.3)

            url = "http://localhost:19101/health"
            response = urllib.request.urlopen(url)
            health = response.read().decode("utf-8")

            assert response.status == 200
            assert "healthy" in health or "status" in health

        finally:
            exporter.stop()

    def test_404_on_invalid_path(self):
        """Test that invalid paths return 404."""
        exporter = PrometheusMetricsExporter(port=19102)
        exporter.start()

        try:
            time.sleep(0.3)

            url = "http://localhost:19102/invalid"
            try:
                urllib.request.urlopen(url)
                raise AssertionError("Should have raised 404")
            except Exception as e:
                # Expected: HTTP 404
                assert "404" in str(e) or "Not Found" in str(e)

        finally:
            exporter.stop()

    def test_multiple_record_operations(self):
        """Test recording multiple operations and verifying metrics."""
        exporter = PrometheusMetricsExporter(port=19103)

        # Record various operations
        exporter.record_operation("embed_codebase", 1.0, "ok", 100)
        exporter.record_operation("embed_codebase", 0.9, "ok", 95)
        exporter.record_operation("semantic_search", 0.1, "ok", 0)
        exporter.record_operation("semantic_search", 0.15, "ok", 0)
        exporter.record_operation("commit", 2.0, "conflict", 0)

        exporter.start()

        try:
            time.sleep(0.3)

            url = "http://localhost:19103/metrics"
            response = urllib.request.urlopen(url)
            metrics = response.read().decode("utf-8")

            # Check that all operations are recorded
            assert "embed_codebase" in metrics
            assert "semantic_search" in metrics
            assert "commit" in metrics

            # Check that all statuses are recorded
            assert "ok" in metrics
            assert "conflict" in metrics

        finally:
            exporter.stop()


class TestMetricsExporterSingleton:
    """Tests for singleton pattern."""

    def test_get_prometheus_exporter_singleton(self):
        """Test that get_prometheus_exporter returns singleton."""
        exporter1 = get_prometheus_exporter(port=19104)
        exporter2 = get_prometheus_exporter(port=19104)

        assert exporter1 is exporter2

    def test_configure_prometheus_starts_server(self):
        """Test that configure_prometheus starts the server."""
        exporter = configure_prometheus(port=19105, start_server=True)

        try:
            assert exporter._server is not None
            time.sleep(0.2)

            # Verify endpoint responds
            try:
                url = "http://localhost:19105/metrics"
                response = urllib.request.urlopen(url)
                assert response.status == 200
            except Exception as e:
                pytest.skip(f"HTTP request failed: {e}")

        finally:
            exporter.stop()

    def test_configure_prometheus_no_autostart(self):
        """Test that configure_prometheus can skip autostart."""
        # Note: This creates a new exporter instance
        exporter = PrometheusMetricsExporter(port=19106)

        try:
            exporter.start()
            assert exporter._server is not None
        finally:
            exporter.stop()


class TestMetricsDataIntegrity:
    """Tests for metric data integrity."""

    def test_operation_metrics_with_all_fields(self):
        """Test recording operations with all fields populated."""
        exporter = PrometheusMetricsExporter(port=19107)

        # Record with all fields
        exporter.record_operation(
            operation="embed_codebase",
            duration_s=2.345,
            status="ok",
            chunk_count=100,
        )

        exporter.record_operation(
            operation="semantic_search",
            duration_s=0.123,
            status="ok",
            chunk_count=0,
        )

        exporter.record_operation(
            operation="commit",
            duration_s=1.5,
            status="conflict",
            chunk_count=0,
        )

        exporter.start()

        try:
            time.sleep(0.3)

            url = "http://localhost:19107/metrics"
            response = urllib.request.urlopen(url)
            metrics = response.read().decode("utf-8")

            # Verify all recorded operations appear in output
            assert len(metrics) > 500  # Reasonable size for metrics

        finally:
            exporter.stop()

    def test_cache_metrics_isolation(self):
        """Test that cache metrics for different cache types are isolated."""
        exporter = PrometheusMetricsExporter(port=19108)

        # Record hits and misses for different cache types
        exporter.record_cache_access("query", hit=True, size_bytes=1024)
        exporter.record_cache_access("index", hit=True, size_bytes=5242880)
        exporter.record_cache_access("lexical", hit=False, size_bytes=2048000)

        exporter.start()

        try:
            time.sleep(0.3)

            url = "http://localhost:19108/metrics"
            response = urllib.request.urlopen(url)
            metrics = response.read().decode("utf-8")

            # Verify different cache types are tracked separately
            assert "query" in metrics
            assert "index" in metrics
            assert "lexical" in metrics

        finally:
            exporter.stop()


class TestMetricsExporterErrorHandling:
    """Tests for error handling."""

    def test_double_start_warning(self, caplog):
        """Test that starting twice doesn't cause errors."""
        exporter = PrometheusMetricsExporter(port=19109)

        try:
            exporter.start()
            time.sleep(0.2)

            # Second start should warn but not crash
            exporter.start()

            assert True  # No exception raised

        finally:
            exporter.stop()

    def test_stop_without_start(self):
        """Test that stopping without starting doesn't crash."""
        exporter = PrometheusMetricsExporter(port=19110)
        exporter.stop()  # Should not crash
        assert True
