"""Health status tracking for buffers."""

import time
from enum import Enum
from typing import Optional
from dataclasses import dataclass

from gigacode.buffer_state import BufferState


class HealthLevel(Enum):
    """Health status level."""
    OK = "ok"
    WARNING = "warning"
    DEGRADED = "degraded"


@dataclass
class HealthStatus:
    """Buffer health status information."""
    buffer_id: str
    state: BufferState
    last_state_change_timestamp: float
    dirty_file_count: int
    index_age_seconds: int
    query_count_since_rebuild: int = 0
    warning_level: HealthLevel = HealthLevel.OK
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "buffer_id": self.buffer_id,
            "state": self.state.value,
            "last_state_change_timestamp": self.last_state_change_timestamp,
            "last_state_change_ago_seconds": time.time() - self.last_state_change_timestamp,
            "dirty_file_count": self.dirty_file_count,
            "index_age_seconds": self.index_age_seconds,
            "query_count_since_rebuild": self.query_count_since_rebuild,
            "warning_level": self.warning_level.value,
            "is_rebuilding": self.state == BufferState.REBUILDING,
            "has_uncommitted_changes": self.state == BufferState.DIRTY,
        }
    
    @staticmethod
    def compute_warning_level(
        dirty_file_count: int,
        index_age_seconds: int,
        dirty_warning_threshold: int = 5,
        dirty_degraded_threshold: int = 20,
        index_warning_seconds: int = 604800,  # 1 week
        index_degraded_seconds: int = 2592000,  # 1 month
    ) -> HealthLevel:
        """Compute warning level based on metrics.
        
        Args:
            dirty_file_count: Number of dirty files
            index_age_seconds: Age of index in seconds
            dirty_warning_threshold: Dirty file count for warning
            dirty_degraded_threshold: Dirty file count for degraded
            index_warning_seconds: Index age for warning
            index_degraded_seconds: Index age for degraded
        
        Returns:
            HealthLevel based on thresholds.
        """
        # Check dirty file thresholds
        if dirty_file_count >= dirty_degraded_threshold:
            return HealthLevel.DEGRADED
        if dirty_file_count >= dirty_warning_threshold:
            return HealthLevel.WARNING
        
        # Check index age thresholds
        if index_age_seconds >= index_degraded_seconds:
            return HealthLevel.DEGRADED
        if index_age_seconds >= index_warning_seconds:
            return HealthLevel.WARNING
        
        return HealthLevel.OK


class HealthStatusTracker:
    """Tracks health status for all buffers."""
    
    def __init__(self):
        """Initialize health tracker."""
        self._health_data: dict[str, dict] = {}
    
    def register_buffer(self, buffer_id: str, state: BufferState) -> None:
        """Register buffer for health tracking.
        
        Args:
            buffer_id: Buffer identifier
            state: Initial buffer state
        """
        self._health_data[buffer_id] = {
            "state": state,
            "last_state_change_timestamp": time.time(),
            "dirty_file_count": 0,
            "index_age_seconds": 0,
            "query_count_since_rebuild": 0,
        }
    
    def update_buffer_state(
        self, 
        buffer_id: str, 
        new_state: BufferState
    ) -> None:
        """Update buffer state in health tracker.
        
        Args:
            buffer_id: Buffer identifier
            new_state: New buffer state
        """
        if buffer_id not in self._health_data:
            self.register_buffer(buffer_id, new_state)
            return
        
        self._health_data[buffer_id]["state"] = new_state
        self._health_data[buffer_id]["last_state_change_timestamp"] = time.time()
        
        # Reset query count on rebuild completion
        if new_state == BufferState.READY:
            self._health_data[buffer_id]["query_count_since_rebuild"] = 0
    
    def update_dirty_file_count(self, buffer_id: str, count: int) -> None:
        """Update dirty file count.
        
        Args:
            buffer_id: Buffer identifier
            count: Number of dirty files
        """
        if buffer_id not in self._health_data:
            return
        self._health_data[buffer_id]["dirty_file_count"] = count
    
    def update_index_age(self, buffer_id: str, age_seconds: int) -> None:
        """Update index age.
        
        Args:
            buffer_id: Buffer identifier
            age_seconds: Age of index in seconds
        """
        if buffer_id not in self._health_data:
            return
        self._health_data[buffer_id]["index_age_seconds"] = age_seconds
    
    def increment_query_count(self, buffer_id: str) -> None:
        """Increment query count since rebuild.
        
        Args:
            buffer_id: Buffer identifier
        """
        if buffer_id not in self._health_data:
            return
        self._health_data[buffer_id]["query_count_since_rebuild"] += 1
    
    def get_health_status(
        self,
        buffer_id: str,
        dirty_warning_threshold: int = 5,
        dirty_degraded_threshold: int = 20,
        index_warning_seconds: int = 604800,
        index_degraded_seconds: int = 2592000,
    ) -> Optional[HealthStatus]:
        """Get health status for a buffer.
        
        Args:
            buffer_id: Buffer identifier
            dirty_warning_threshold: Threshold for dirty file warning
            dirty_degraded_threshold: Threshold for dirty file degraded
            index_warning_seconds: Threshold for index age warning
            index_degraded_seconds: Threshold for index age degraded
        
        Returns:
            HealthStatus or None if buffer not tracked.
        """
        if buffer_id not in self._health_data:
            return None
        
        data = self._health_data[buffer_id]
        dirty_count = data["dirty_file_count"]
        index_age = data["index_age_seconds"]
        
        warning_level = HealthStatus.compute_warning_level(
            dirty_count,
            index_age,
            dirty_warning_threshold,
            dirty_degraded_threshold,
            index_warning_seconds,
            index_degraded_seconds,
        )
        
        return HealthStatus(
            buffer_id=buffer_id,
            state=data["state"],
            last_state_change_timestamp=data["last_state_change_timestamp"],
            dirty_file_count=dirty_count,
            index_age_seconds=index_age,
            query_count_since_rebuild=data["query_count_since_rebuild"],
            warning_level=warning_level,
        )
    
    def get_all_health_statuses(
        self,
        dirty_warning_threshold: int = 5,
        dirty_degraded_threshold: int = 20,
        index_warning_seconds: int = 604800,
        index_degraded_seconds: int = 2592000,
    ) -> dict[str, HealthStatus]:
        """Get health status for all buffers.
        
        Returns:
            Dictionary mapping buffer_id to HealthStatus.
        """
        return {
            buffer_id: self.get_health_status(
                buffer_id,
                dirty_warning_threshold,
                dirty_degraded_threshold,
                index_warning_seconds,
                index_degraded_seconds,
            )
            for buffer_id in self._health_data
            if self.get_health_status(
                buffer_id,
                dirty_warning_threshold,
                dirty_degraded_threshold,
                index_warning_seconds,
                index_degraded_seconds,
            )
            is not None
        }
