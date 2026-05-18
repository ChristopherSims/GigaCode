"""
Undo Stack with Branching

Provides stack-based undo/redo with git-like branching for experimental work.
Users can try different approaches and switch between them without full rollback.

Key features:
- Per-operation undo/redo stack
- Named branches for experiments
- Branch switching and merging
- State snapshots and restoration
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from gigacode.metrics import log_metric


class OperationType(Enum):
    """Types of undoable operations."""

    WRITE_CODE = "write_code"
    DELETE_CODE = "delete_code"
    RENAME_FILE = "rename_file"
    RUN_TESTS = "run_tests"
    COMMIT = "commit"


@dataclass
class UndoableOperation:
    """Single undoable operation."""

    id: str
    operation_type: OperationType
    timestamp: datetime
    file: Optional[str]
    original_content: Optional[str]
    new_content: Optional[str]
    description: str
    reversible: bool = True

    def reverse(self) -> "UndoableOperation":
        """Create reverse operation."""
        return UndoableOperation(
            id=str(uuid.uuid4()),
            operation_type=self.operation_type,
            timestamp=datetime.now(),
            file=self.file,
            original_content=self.new_content,
            new_content=self.original_content,
            description=f"Undo: {self.description}",
            reversible=self.reversible,
        )


@dataclass
class BranchSnapshot:
    """Snapshot of buffer state at branch point."""

    branch_name: str
    parent_branch: str
    created_at: datetime
    operations: List[UndoableOperation] = field(default_factory=list)
    file_states: Dict[str, str] = field(default_factory=dict)
    description: str = ""


class UndoRedoStack:
    """Manages undo/redo operations."""

    def __init__(self, max_stack_size: int = 100):
        """
        Initialize stack.

        Args:
            max_stack_size: Max operations to keep in history
        """
        self.undo_stack: List[UndoableOperation] = []
        self.redo_stack: List[UndoableOperation] = []
        self.max_stack_size = max_stack_size

    def push(self, operation: UndoableOperation) -> None:
        """Push operation onto undo stack."""
        self.undo_stack.append(operation)
        self.redo_stack.clear()  # Clear redo when new action taken

        # Trim old operations if exceeding size
        if len(self.undo_stack) > self.max_stack_size:
            self.undo_stack.pop(0)

    def undo(self) -> Optional[UndoableOperation]:
        """Pop from undo stack and move the operation to redo stack."""
        if not self.undo_stack:
            return None

        operation = self.undo_stack.pop()
        self.redo_stack.append(operation)

        return operation

    def redo(self) -> Optional[UndoableOperation]:
        """Pop from redo stack and move the operation back to undo stack."""
        if not self.redo_stack:
            return None

        operation = self.redo_stack.pop()
        self.undo_stack.append(operation)

        return operation

    def peek_undo(self) -> Optional[UndoableOperation]:
        """Peek at next undo operation without removing."""
        return self.undo_stack[-1] if self.undo_stack else None

    def peek_redo(self) -> Optional[UndoableOperation]:
        """Peek at next redo operation without removing."""
        return self.redo_stack[-1] if self.redo_stack else None

    def get_history(self) -> List[UndoableOperation]:
        """Get full undo history."""
        return list(self.undo_stack)


class BranchManager:
    """Manages buffer branches for experimental work."""

    def __init__(self, default_branch: str = "main"):
        """
        Initialize branch manager.

        Args:
            default_branch: Default branch name
        """
        self.branches: Dict[str, BranchSnapshot] = {}
        self.current_branch = default_branch
        self.undo_stacks: Dict[str, UndoRedoStack] = {default_branch: UndoRedoStack()}

    def create_branch(
        self,
        branch_name: str,
        description: str = "",
    ) -> BranchSnapshot:
        """
        Create new branch from current state.

        Args:
            branch_name: Name for new branch
            description: Optional description

        Returns:
            BranchSnapshot for new branch
        """
        if branch_name in self.branches:
            raise ValueError(f"Branch '{branch_name}' already exists")

        snapshot = BranchSnapshot(
            branch_name=branch_name,
            parent_branch=self.current_branch,
            created_at=datetime.now(),
            description=description,
        )

        self.branches[branch_name] = snapshot
        self.undo_stacks[branch_name] = UndoRedoStack()

        log_metric(
            "branch_manager.create_branch",
            {
                "branch": branch_name,
                "parent": self.current_branch,
            },
        )

        return snapshot

    def checkout_branch(self, branch_name: str) -> None:
        """
        Switch to a different branch.

        Args:
            branch_name: Branch to switch to
        """
        if branch_name not in self.branches and branch_name != "main":
            raise ValueError(f"Branch '{branch_name}' does not exist")

        self.current_branch = branch_name
        log_metric("branch_manager.checkout", {"branch": branch_name})

    def delete_branch(self, branch_name: str) -> None:
        """Delete a branch."""
        if branch_name == "main":
            raise ValueError("Cannot delete main branch")

        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist")

        del self.branches[branch_name]
        del self.undo_stacks[branch_name]

    def list_branches(self) -> List[Dict]:
        """List all branches."""
        branches_list = []

        for name, snapshot in self.branches.items():
            branches_list.append(
                {
                    "name": name,
                    "parent": snapshot.parent_branch,
                    "created": snapshot.created_at.isoformat(),
                    "description": snapshot.description,
                    "operations": len(snapshot.operations),
                    "current": name == self.current_branch,
                }
            )

        return branches_list

    def get_undo_stack(self, branch: Optional[str] = None) -> UndoRedoStack:
        """Get undo/redo stack for branch."""
        target_branch = branch or self.current_branch
        return self.undo_stacks.get(target_branch, UndoRedoStack())


class UndoableBufferOperations:
    """Wraps buffer operations with undo/redo support."""

    def __init__(self, buffer_manager, undo_stack: UndoRedoStack):
        """
        Initialize.

        Args:
            buffer_manager: BufferManager instance
            undo_stack: UndoRedoStack for tracking
        """
        self.buffer_manager = buffer_manager
        self.undo_stack = undo_stack

    def write_code(
        self,
        buffer_id: str,
        file: str,
        new_content: str,
    ) -> str:
        """
        Write code with undo support.

        Args:
            buffer_id: Target buffer
            file: File to write
            new_content: New file content

        Returns:
            Operation ID
        """
        # Read original content - adapt to actual API
        original = ""
        if hasattr(self.buffer_manager, "read_code"):
            result = self.buffer_manager.read_code(buffer_id, file=file)
            if isinstance(result, dict) and result.get("status") == "ok":
                lines = result.get("lines", [])
                original = "\n".join(lines) if isinstance(lines, list) else str(lines)
            else:
                original = str(result)
        elif hasattr(self.buffer_manager, "read_file"):
            original = self.buffer_manager.read_file(buffer_id, file)
        else:
            # Fallback: try to get from snapshot
            original = ""

        # Perform write - adapt to actual write_code signature (start_line, new_lines)
        new_lines = new_content.splitlines() if isinstance(new_content, str) else list(new_content)
        if hasattr(self.buffer_manager, "write_code"):
            # Actual API: write_code(buffer_id, file, start_line, new_lines, end_line=None)
            result = self.buffer_manager.write_code(
                buffer_id,
                file,
                1,
                new_lines,
                end_line=None,
            )
        else:
            result = {"status": "error", "message": "write_code not available"}

        # Create undoable operation
        operation = UndoableOperation(
            id=str(uuid.uuid4()),
            operation_type=OperationType.WRITE_CODE,
            timestamp=datetime.now(),
            file=file,
            original_content=original,
            new_content=new_content,
            description=f"Modified {file}",
        )

        self.undo_stack.push(operation)

        log_metric("undoable_operations.write_code", {"file": file})

        return operation.id

    def delete_file(self, buffer_id: str, file: str) -> str:
        """Delete file with undo support."""
        # Read content before deletion - adapt to actual API
        original = ""
        if hasattr(self.buffer_manager, "read_code"):
            result = self.buffer_manager.read_code(buffer_id, file=file)
            if isinstance(result, dict) and result.get("status") == "ok":
                lines = result.get("lines", [])
                original = "\n".join(lines) if isinstance(lines, list) else str(lines)
            else:
                original = str(result)
        elif hasattr(self.buffer_manager, "read_file"):
            original = self.buffer_manager.read_file(buffer_id, file)

        # Perform delete
        if hasattr(self.buffer_manager, "delete_file"):
            self.buffer_manager.delete_file(buffer_id, file)

        # Create undoable operation
        operation = UndoableOperation(
            id=str(uuid.uuid4()),
            operation_type=OperationType.DELETE_CODE,
            timestamp=datetime.now(),
            file=file,
            original_content=original,
            new_content=None,
            description=f"Deleted {file}",
        )

        self.undo_stack.push(operation)

        return operation.id


class BranchedBufferManager:
    """Buffer manager with branching support."""

    def __init__(self, buffer_manager):
        """Initialize manager."""
        self.buffer_manager = buffer_manager
        self.branch_managers: Dict[str, BranchManager] = {}

    def _get_branch_manager(self, buffer_id: str) -> BranchManager:
        """Get or create branch manager for buffer."""
        if buffer_id not in self.branch_managers:
            self.branch_managers[buffer_id] = BranchManager()
        return self.branch_managers[buffer_id]

    @staticmethod
    def _split_content(content: Optional[str]) -> List[str]:
        """Convert stored file content into write_code-compatible lines."""
        if not content:
            return []
        return content.splitlines()

    def _apply_operation_content(
        self,
        buffer_id: str,
        operation: UndoableOperation,
        content: Optional[str],
    ) -> None:
        """Apply stored full-file content for an operation."""
        if not operation.file or not hasattr(self.buffer_manager, "write_code"):
            return

        self.buffer_manager.write_code(
            buffer_id,
            operation.file,
            1,
            self._split_content(content),
            end_line=None,
        )

    def record_write(
        self,
        buffer_id: str,
        file: str,
        original_content: str,
        new_content: str,
        description: str,
    ) -> str:
        """Record a write_code operation for later undo/redo."""
        branch_mgr = self._get_branch_manager(buffer_id)
        stack = branch_mgr.get_undo_stack()
        operation = UndoableOperation(
            id=str(uuid.uuid4()),
            operation_type=OperationType.WRITE_CODE,
            timestamp=datetime.now(),
            file=file,
            original_content=original_content,
            new_content=new_content,
            description=description,
        )
        stack.push(operation)
        return operation.id

    def undo(self, buffer_id: str, steps: int = 1) -> int:
        """
        Undo last N operations.

        Args:
            buffer_id: Target buffer
            steps: Number of steps to undo
        """
        branch_mgr = self._get_branch_manager(buffer_id)
        stack = branch_mgr.get_undo_stack()
        undone = 0

        for _ in range(steps):
            operation = stack.undo()
            if not operation:
                break

            if operation.operation_type == OperationType.WRITE_CODE:
                self._apply_operation_content(buffer_id, operation, operation.original_content)
                undone += 1

        log_metric(
            "branched_buffer.undo",
            {
                "buffer": buffer_id,
                "steps": undone,
            },
        )

        return undone

    def redo(self, buffer_id: str, steps: int = 1) -> int:
        """Redo last N undone operations."""
        branch_mgr = self._get_branch_manager(buffer_id)
        stack = branch_mgr.get_undo_stack()
        redone = 0

        for _ in range(steps):
            operation = stack.redo()
            if not operation:
                break

            if operation.operation_type == OperationType.WRITE_CODE:
                self._apply_operation_content(buffer_id, operation, operation.new_content)
                redone += 1

        log_metric(
            "branched_buffer.redo",
            {
                "buffer": buffer_id,
                "steps": redone,
            },
        )

        return redone

    def branch(
        self,
        buffer_id: str,
        branch_name: str,
        description: str = "",
    ) -> Dict:
        """
        Create new branch.

        Args:
            buffer_id: Target buffer
            branch_name: Name for new branch
            description: Optional description

        Returns:
            Branch info
        """
        branch_mgr = self._get_branch_manager(buffer_id)
        snapshot = branch_mgr.create_branch(branch_name, description)

        return {
            "branch": branch_name,
            "parent": snapshot.parent_branch,
            "created": snapshot.created_at.isoformat(),
        }

    def checkout(self, buffer_id: str, branch_name: str) -> None:
        """Switch to branch."""
        branch_mgr = self._get_branch_manager(buffer_id)
        branch_mgr.checkout_branch(branch_name)

    def list_branches(self, buffer_id: str) -> List[Dict]:
        """List all branches for buffer."""
        branch_mgr = self._get_branch_manager(buffer_id)
        return branch_mgr.list_branches()

    def delete_branch(self, buffer_id: str, branch_name: str) -> None:
        """Delete a branch."""
        branch_mgr = self._get_branch_manager(buffer_id)
        branch_mgr.delete_branch(branch_name)

    def get_history(self, buffer_id: str) -> List[Dict]:
        """Get edit history for current branch."""
        branch_mgr = self._get_branch_manager(buffer_id)
        stack = branch_mgr.get_undo_stack()

        return [
            {
                "id": op.id,
                "type": op.operation_type.value,
                "file": op.file,
                "timestamp": op.timestamp.isoformat(),
                "description": op.description,
            }
            for op in stack.get_history()
        ]


class UndoRedoService:
    """Public interface for undo/redo with branching."""

    def __init__(self, branched_buffer_manager: BranchedBufferManager):
        """Initialize service."""
        self.branched_buffer_manager = branched_buffer_manager

    def undo(self, buffer_id: str, steps: int = 1) -> int:
        """Undo operations."""
        return self.branched_buffer_manager.undo(buffer_id, steps)

    def redo(self, buffer_id: str, steps: int = 1) -> int:
        """Redo operations."""
        return self.branched_buffer_manager.redo(buffer_id, steps)

    def record_write(
        self,
        buffer_id: str,
        file: str,
        original_content: str,
        new_content: str,
        description: str,
    ) -> str:
        """Record a write operation for undo/redo."""
        return self.branched_buffer_manager.record_write(
            buffer_id=buffer_id,
            file=file,
            original_content=original_content,
            new_content=new_content,
            description=description,
        )

    def branch(
        self,
        buffer_id: str,
        name: str,
        description: str = "",
    ) -> Dict:
        """Create branch."""
        return self.branched_buffer_manager.branch(buffer_id, name, description)

    def checkout(self, buffer_id: str, branch: str) -> None:
        """Checkout branch."""
        self.branched_buffer_manager.checkout(buffer_id, branch)

    def list_branches(self, buffer_id: str) -> List[Dict]:
        """List branches."""
        return self.branched_buffer_manager.list_branches(buffer_id)

    def delete_branch(self, buffer_id: str, branch: str) -> None:
        """Delete branch."""
        self.branched_buffer_manager.delete_branch(buffer_id, branch)

    def history(self, buffer_id: str) -> List[Dict]:
        """Get edit history."""
        return self.branched_buffer_manager.get_history(buffer_id)
