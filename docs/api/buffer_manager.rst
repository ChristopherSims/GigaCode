Buffer Manager
===============

``gigacode.buffer_manager.BufferManager`` manages code buffers and their lifecycle.

Overview
~~~~~~~~

The BufferManager is responsible for:

- Creating and managing code buffers
- Storing buffer metadata and state
- Tracking file changes
- Managing buffer disk storage
- Garbage collection of old buffers

Core Class
~~~~~~~~~~

.. autoclass:: gigacode.buffer_manager.BufferManager
   :members:
   :undoc-members:
   :show-inheritance:

Key Methods
~~~~~~~~~~~

**Buffer Creation**

.. code-block:: python

    buffer_id = manager.create_buffer(
        buffer_id="my_project",
        metadata={
            "source": "/path/to/project",
            "language": "python"
        }
    )

**Buffer Retrieval**

.. code-block:: python

    buffer = manager.get_buffer(buffer_id="my_project")
    if buffer:
        print(f"Files: {buffer.file_count}")

**Buffer Update**

.. code-block:: python

    manager.update_buffer(
        buffer_id="my_project",
        metadata={"last_updated": datetime.now()}
    )

**List Buffers**

.. code-block:: python

    buffers = manager.list_buffers()
    for buffer_id, metadata in buffers.items():
        print(f"{buffer_id}: {metadata['file_count']} files")

**Delete Buffer**

.. code-block:: python

    manager.delete_buffer(buffer_id="my_project")

State Management
~~~~~~~~~~~~~~~~

BufferManager tracks buffer state:

.. code-block:: python

    # Get buffer state
    state = manager.get_buffer_state(buffer_id="my_project")
    print(f"State: {state.status}")  # active, inactive, error
    
    # Update state
    manager.set_buffer_state(
        buffer_id="my_project",
        status="active",
        metadata={"last_search": datetime.now()}
    )

File Tracking
~~~~~~~~~~~~~

Track files within buffers:

.. code-block:: python

    # Get files in buffer
    files = manager.get_buffer_files(buffer_id="my_project")
    for file_info in files:
        print(f"{file_info['path']}: {file_info['size']} bytes")
    
    # Track file changes
    changed_files = manager.get_changed_files(buffer_id="my_project")
    for file_path in changed_files:
        print(f"Modified: {file_path}")

Storage Management
~~~~~~~~~~~~~~~~~~

BufferManager handles persistent storage:

.. code-block:: python

    # Get storage location
    storage_path = manager.get_buffer_storage(buffer_id="my_project")
    print(f"Stored at: {storage_path}")
    
    # Get buffer size
    size_mb = manager.get_buffer_size(buffer_id="my_project")
    print(f"Size: {size_mb:.2f} MB")
    
    # Cleanup old data
    manager.cleanup_buffer_storage(buffer_id="my_project")

Configuration
~~~~~~~~~~~~~

.. code-block:: python

    from gigacode.buffer_manager import BufferManager
    
    manager = BufferManager(
        work_dir="/path/to/work",
        max_buffers=10,
        auto_cleanup=True,
        cleanup_interval=3600
    )

Example: Complete Buffer Lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gigacode.buffer_manager import BufferManager
    
    manager = BufferManager()
    
    # 1. Create buffer
    buffer_id = manager.create_buffer(
        buffer_id="my_project",
        metadata={"source": "/path/to/project"}
    )
    print(f"✓ Created: {buffer_id}")
    
    # 2. Get buffer info
    buffer = manager.get_buffer(buffer_id)
    print(f"✓ Files: {buffer.file_count}")
    
    # 3. Update metadata
    manager.update_buffer(
        buffer_id,
        metadata={"description": "My project"}
    )
    
    # 4. Track changes
    changed = manager.get_changed_files(buffer_id)
    print(f"✓ Changed files: {len(changed)}")
    
    # 5. Cleanup and delete
    manager.cleanup_buffer_storage(buffer_id)
    manager.delete_buffer(buffer_id)
    print(f"✓ Deleted: {buffer_id}")

See Also
~~~~~~~~

- :class:`gigacode.buffer_state.BufferState` - Buffer state management
- :class:`gigacode.file_watcher.FileWatcher` - File change detection
- :doc:`search_service` - Search using buffers
