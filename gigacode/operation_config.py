"""
Operation Configuration Module.

This module defines settings and constraints for different types of operations
(query, write, read, rebuild) on code buffers. It controls when operations are
allowed based on the buffer's current state and provides health thresholds.

Key Concepts:
    - OperationType: Categories of operations (QUERY, WRITE, READ, REBUILD)
    - State Requirements: Which buffer states allow each operation type
    - Health Thresholds: Warning/degraded levels for dirty files and index age

Example:
    >>> config = OperationConfig
    >>> requirements = config.get_state_requirements()
    >>> allowed_states = requirements[OperationType.QUERY]
    >>> # QUERY operations allowed when buffer is READY or DIRTY
"""

from enum import Enum


class OperationType(Enum):
    """
    > Classification of different operation types.
    >
    > Each operation type has specific rules about which buffer states it can run in.
    > For example, WRITE operations only work on READY buffers, but QUERY operations
    > work on both READY and DIRTY buffers.
    >
    > Attributes:
    >     QUERY: Search or lookup operations (fast, read-only)
    >     WRITE: Modify code in a buffer (requires READY state)
    >     READ: Read code without searching (fast, read-only)
    >     REBUILD: Rebuild the search index (can work on both READY and DIRTY)
    """
    QUERY = "query"      # Search operations
    WRITE = "write"      # Code write operations
    REBUILD = "rebuild"  # Index rebuild operations
    READ = "read"        # Code read operations


class OperationConfig:
    """
    > Configuration Settings for Code Buffer Operations.
    >
    > This class holds all the settings that control how operations work on buffers.
    > It acts like a control panel where you can adjust timeouts, health thresholds,
    > and enable/disable different safeguards.
    >
    > Example Configuration:
    >     - QUERY_TIMEOUT_SECONDS = 30: A query can't run longer than 30 seconds
    >     - DIRTY_FILE_WARNING_THRESHOLD = 5: Warn if 5+ files have uncommitted changes
    >     - INDEX_AGE_DEGRADED_SECONDS = 30 days: Index is degraded after 30 days
    >
    > State Guards:
    >     - Prevent invalid operations based on buffer state
    >     - Example: Block WRITE operations on DIRTY buffers
    >     - Can be enabled/disabled as needed
    """
    
    # Query behavior in DIRTY state
    BLOCK_DIRTY_QUERIES = False  # Default: warning only
    
    # Query timeouts
    QUERY_TIMEOUT_SECONDS = 30
    
    # Rebuild coordination
    MAX_PENDING_OPERATIONS = 100
    OPERATION_RETRY_ATTEMPTS = 3
    PENDING_OPERATION_TIMEOUT_SECONDS = 300  # 5 minutes
    
    # Health thresholds
    DIRTY_FILE_WARNING_THRESHOLD = 5
    DIRTY_FILE_DEGRADED_THRESHOLD = 20
    INDEX_AGE_WARNING_SECONDS = 7 * 24 * 3600  # 1 week
    INDEX_AGE_DEGRADED_SECONDS = 30 * 24 * 3600  # 1 month
    
    # Enable/disable state guards
    ENABLE_QUERY_STATE_GUARDS = True
    ENABLE_WRITE_STATE_GUARDS = True
    ENABLE_HEALTH_TRACKING = True
    
    @staticmethod
    def get_state_requirements() -> dict:
        """
        > Get the allowed buffer states for each operation type.
        >
        > This method returns a dictionary that maps each operation type to the
        > list of buffer states where that operation is allowed to run.
        >
        > Returns:
        >     dict: Maps OperationType to list of allowed BufferState values
        >
        > Example:
        >     >>> requirements = OperationConfig.get_state_requirements()
        >     >>> query_states = requirements[OperationType.QUERY]
        >     >>> # query_states = [BufferState.READY, BufferState.DIRTY]
        >     >>> # This means queries work on both READY and DIRTY buffers
        >
        > State Rules:
        >     - QUERY: Works on READY (fresh) and DIRTY (has changes) buffers
        >     - READ: Works on READY and DIRTY buffers (like QUERY)
        >     - WRITE: Works ONLY on READY buffers (must rebuild before writing)
        >     - REBUILD: Works on READY and DIRTY buffers
        """
        from gigacode.buffer_state import BufferState
        
        return {
            OperationType.QUERY: [BufferState.READY, BufferState.DIRTY],
            OperationType.READ: [BufferState.READY, BufferState.DIRTY],
            OperationType.WRITE: [BufferState.READY],
            OperationType.REBUILD: [BufferState.READY, BufferState.DIRTY],
        }
