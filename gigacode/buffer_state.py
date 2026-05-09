"""Buffer state machine for lifecycle management.

Defines explicit states and valid transitions:
- READY: Buffer ready for operations (no pending changes)
- DIRTY: Buffer has modified files (pending commit/discard)
- REBUILDING: Buffer indices are being rebuilt

Valid transitions:
- READY → DIRTY (when file is written)
- DIRTY → READY (when changes committed/discarded)
- DIRTY/READY → REBUILDING (during index rebuild)
- REBUILDING → READY (when rebuild completes)
"""

from enum import Enum
from typing import Optional


__all__ = [
    "BufferState",
    "BufferStateTransition",
]


class BufferState(Enum):
    """Buffer state enumeration."""
    
    READY = "ready"
    DIRTY = "dirty"
    REBUILDING = "rebuilding"
    
    def __str__(self) -> str:
        return self.value


class BufferStateTransition:
    """Manages buffer state transitions with validation."""
    
    # Valid state transitions: from_state -> list of valid next states
    VALID_TRANSITIONS = {
        BufferState.READY: [BufferState.DIRTY, BufferState.REBUILDING],
        BufferState.DIRTY: [BufferState.READY, BufferState.REBUILDING],
        BufferState.REBUILDING: [BufferState.READY],
    }
    
    @classmethod
    def is_valid(cls, from_state: BufferState, to_state: BufferState) -> bool:
        """Check if transition is valid.
        
        Args:
            from_state: Current state
            to_state: Desired state
            
        Returns:
            True if transition is allowed
        """
        return to_state in cls.VALID_TRANSITIONS.get(from_state, [])
    
    @classmethod
    def validate_or_raise(cls, from_state: BufferState, to_state: BufferState) -> None:
        """Validate transition or raise ValueError.
        
        Args:
            from_state: Current state
            to_state: Desired state
            
        Raises:
            ValueError: If transition is invalid
        """
        if not cls.is_valid(from_state, to_state):
            valid = cls.VALID_TRANSITIONS.get(from_state, [])
            valid_str = ", ".join(str(s) for s in valid)
            raise ValueError(
                f"Invalid state transition: {from_state} → {to_state}. "
                f"Valid transitions: {valid_str}"
            )
    
    @classmethod
    def describe(cls) -> str:
        """Return description of state machine.
        
        Returns:
            String describing all valid transitions
        """
        lines = ["Buffer State Machine:"]
        for state, next_states in cls.VALID_TRANSITIONS.items():
            next_str = ", ".join(str(s) for s in next_states)
            lines.append(f"  {state} → {next_str}")
        return "\n".join(lines)
