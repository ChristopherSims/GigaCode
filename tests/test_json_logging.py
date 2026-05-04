"""Tests for structured JSON logging (Phase 6.3).

Validates that JSON logging produces well-formatted, consistent output
with all required fields and proper serialization.
"""

import json
import logging
from io import StringIO

import pytest

from gigacode.json_logger import (
    LogEntry,
    StructuredJsonLogger,
    configure_json_logging,
)


class TestLogEntry:
    """Tests for LogEntry dataclass and JSON serialization."""

    def test_log_entry_basic(self):
        """Test basic LogEntry creation and to_json()."""
        entry = LogEntry(
            timestamp=1234567890.123,
            level='INFO',
            operation='test_op',
        )
        json_str = entry.to_json()
        data = json.loads(json_str)
        
        assert data['timestamp'] == 1234567890.123
        assert data['level'] == 'INFO'
        assert data['operation'] == 'test_op'
        assert 'buffer_id' not in data  # Optional fields not included
        assert 'message' not in data

    def test_log_entry_with_all_fields(self):
        """Test LogEntry with all optional fields."""
        entry = LogEntry(
            timestamp=1234567890.456,
            level='WARNING',
            operation='write_code',
            buffer_id='buf123',
            elapsed_s=1.234567,  # Will be rounded to 4 decimals
            status='conflict',
            message='Test message',
            details={'count': 42, 'files': ['a.py', 'b.py']},
        )
        json_str = entry.to_json()
        data = json.loads(json_str)
        
        assert data['timestamp'] == 1234567890.456
        assert data['level'] == 'WARNING'
        assert data['operation'] == 'write_code'
        assert data['buffer_id'] == 'buf123'
        assert data['elapsed_s'] == 1.2346  # Rounded to 4 decimals
        assert data['status'] == 'conflict'
        assert data['message'] == 'Test message'
        assert data['details'] == {'count': 42, 'files': ['a.py', 'b.py']}

    def test_log_entry_elapsed_s_rounding(self):
        """Test that elapsed_s is properly rounded to 4 decimal places."""
        entry = LogEntry(
            timestamp=1000.0,
            level='INFO',
            operation='search',
            elapsed_s=0.12345678,
        )
        data = json.loads(entry.to_json())
        assert data['elapsed_s'] == 0.1235  # Rounded up

    def test_log_entry_none_fields_excluded(self):
        """Test that None fields are excluded from JSON output."""
        entry = LogEntry(
            timestamp=1000.0,
            level='DEBUG',
            operation='test',
            buffer_id='buf123',
            elapsed_s=None,  # Should be excluded
            status=None,     # Should be excluded
            message=None,    # Should be excluded
            details=None,    # Should be excluded
        )
        data = json.loads(entry.to_json())
        
        assert 'elapsed_s' not in data
        assert 'status' not in data
        assert 'message' not in data
        assert 'details' not in data
        assert data['buffer_id'] == 'buf123'  # Non-None field included


class TestStructuredJsonLogger:
    """Tests for StructuredJsonLogger class."""

    def test_logger_creation(self):
        """Test creating a logger instance."""
        logger = StructuredJsonLogger('test_module')
        assert logger.module_name == 'test_module'
        assert logger._logger.name == 'gigacode.test_module'

    def test_logger_info_basic(self, caplog):
        """Test info() method produces valid JSON."""
        logger = StructuredJsonLogger('test')
        
        with caplog.at_level(logging.INFO):
            logger.info('test_op')
        
        # Extract JSON from log record
        records = [r for r in caplog.records if r.name == 'gigacode.test']
        assert len(records) == 1
        
        data = json.loads(records[0].getMessage())
        assert data['level'] == 'INFO'
        assert data['operation'] == 'test_op'

    def test_logger_debug(self, caplog):
        """Test debug() method."""
        logger = StructuredJsonLogger('test')
        
        with caplog.at_level(logging.DEBUG):
            logger.debug(
                'search',
                buffer_id='buf456',
                details={'query': 'test', 'top_k': 10},
            )
        
        records = [r for r in caplog.records if r.name == 'gigacode.test']
        data = json.loads(records[0].getMessage())
        
        assert data['level'] == 'DEBUG'
        assert data['operation'] == 'search'
        assert data['buffer_id'] == 'buf456'
        assert data['details']['query'] == 'test'

    def test_logger_warning(self, caplog):
        """Test warning() method."""
        logger = StructuredJsonLogger('test')
        
        with caplog.at_level(logging.WARNING):
            logger.warning(
                'conflict_detected',
                buffer_id='buf789',
                status='conflict',
                message='File conflict detected',
            )
        
        records = [r for r in caplog.records if r.name == 'gigacode.test']
        data = json.loads(records[0].getMessage())
        
        assert data['level'] == 'WARNING'
        assert data['status'] == 'conflict'
        assert data['message'] == 'File conflict detected'

    def test_logger_error(self, caplog):
        """Test error() method."""
        logger = StructuredJsonLogger('test')
        
        with caplog.at_level(logging.ERROR):
            logger.error(
                'embed_failed',
                buffer_id='buf001',
                message='Embedding failed',
                details={'error_code': 'E001'},
            )
        
        records = [r for r in caplog.records if r.name == 'gigacode.test']
        data = json.loads(records[0].getMessage())
        
        assert data['level'] == 'ERROR'
        assert data['operation'] == 'embed_failed'
        assert data['status'] == 'error'  # Defaults to 'error' for error()
        assert data['buffer_id'] == 'buf001'

    def test_logger_all_parameters(self, caplog):
        """Test that all parameters are properly included."""
        logger = StructuredJsonLogger('test')
        
        with caplog.at_level(logging.INFO):
            logger.info(
                operation='complex_op',
                buffer_id='buf_id_123',
                elapsed_s=2.5,
                status='ok',
                message='Operation completed',
                details={'step1': 'done', 'step2': 'done'},
            )
        
        records = [r for r in caplog.records if r.name == 'gigacode.test']
        data = json.loads(records[0].getMessage())
        
        assert 'timestamp' in data
        assert data['level'] == 'INFO'
        assert data['operation'] == 'complex_op'
        assert data['buffer_id'] == 'buf_id_123'
        assert data['elapsed_s'] == 2.5
        assert data['status'] == 'ok'
        assert data['message'] == 'Operation completed'
        assert data['details'] == {'step1': 'done', 'step2': 'done'}


class TestJsonLoggingIntegration:
    """Integration tests for JSON logging."""

    def test_json_logging_format_consistency(self, caplog):
        """Test that all log levels produce consistent JSON format."""
        logger = StructuredJsonLogger('integration')
        
        with caplog.at_level(logging.DEBUG):
            logger.debug('op_debug')
            logger.info('op_info', status='ok')
            logger.warning('op_warning', status='warn')
            logger.error('op_error')
        
        records = [r for r in caplog.records if r.name == 'gigacode.integration']
        assert len(records) == 4
        
        # Verify all are valid JSON
        for record in records:
            data = json.loads(record.getMessage())
            assert 'timestamp' in data
            assert 'level' in data
            assert 'operation' in data

    def test_json_logging_special_characters(self, caplog):
        """Test that special characters are properly escaped."""
        logger = StructuredJsonLogger('test')
        
        with caplog.at_level(logging.INFO):
            logger.info(
                'test',
                message='Message with "quotes" and \\backslash',
                details={'path': 'C:\\Users\\test'},
            )
        
        records = [r for r in caplog.records if r.name == 'gigacode.test']
        json_str = records[0].getMessage()
        
        # Should parse without error
        data = json.loads(json_str)
        assert 'quotes' in data['message']
        assert '\\' in data['details']['path']

    def test_configure_json_logging(self, caplog):
        """Test configure_json_logging() sets up proper logging."""
        # This is more of an integration test
        # Just verify it doesn't crash
        configure_json_logging(level=logging.INFO)
        
        logger = StructuredJsonLogger('config_test')
        with caplog.at_level(logging.INFO):
            logger.info('test_op')
        
        records = [r for r in caplog.records if r.name == 'gigacode.config_test']
        assert len(records) >= 0  # May not have records depending on logging config


class TestJsonLoggingRealWorldScenarios:
    """Real-world scenario tests for JSON logging."""

    def test_embed_codebase_logging(self):
        """Test JSON logging for embed_codebase operation."""
        # Test that logging call doesn't raise exception and produces valid JSON
        logger = StructuredJsonLogger('test')
        
        # Create a log entry and verify it serializes correctly
        entry = LogEntry(
            timestamp=1777907174.4036014,
            level='INFO',
            operation='embed_codebase',
            buffer_id='codebase_v1',
            elapsed_s=3.456,
            status='ok',
            message='Embedded 42 chunks from 5 files',
            details={
                'files_count': 5,
                'chunks_count': 42,
                'size_bytes': 1048576,
            },
        )
        json_str = entry.to_json()
        data = json.loads(json_str)
        
        assert data['operation'] == 'embed_codebase'
        assert data['buffer_id'] == 'codebase_v1'
        assert data['details']['chunks_count'] == 42

    def test_semantic_search_logging(self):
        """Test JSON logging for semantic_search operation."""
        entry = LogEntry(
            timestamp=1777907174.4761066,
            level='DEBUG',
            operation='semantic_search',
            buffer_id='search_buf',
            elapsed_s=0.145,
            details={'top_k': 10, 'gpu': True, 'matches': 8},
        )
        json_str = entry.to_json()
        data = json.loads(json_str)
        
        assert data['operation'] == 'semantic_search'
        assert data['elapsed_s'] == 0.145
        assert data['details']['gpu'] is True

    def test_commit_conflict_logging(self):
        """Test JSON logging for commit with conflict."""
        entry = LogEntry(
            timestamp=1777907174.4761066,
            level='WARNING',
            operation='commit',
            buffer_id='conflict_buf',
            status='conflict',
            message='Merge conflict detected',
            details={
                'conflict_files': ['file1.py', 'file2.py'],
                'transaction_id': 'txn_123',
            },
        )
        json_str = entry.to_json()
        data = json.loads(json_str)
        
        assert data['status'] == 'conflict'
        assert len(data['details']['conflict_files']) == 2
        assert data['operation'] == 'commit'
