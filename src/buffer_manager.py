"""Vulkan buffer allocation, upload, download, and patching.

When the Vulkan bindings are unavailable the manager transparently falls back
to CPU-side numpy arrays so that the agent tool can still be exercised.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.vulkan_context import VulkanContext, _HAS_VULKAN, vk

logger = logging.getLogger(__name__)


@dataclass
class ManagedBuffer:
    """Handle for a buffer that may live on GPU or CPU."""

    name: str
    size_bytes: int
    # Vulkan handles (None in CPU mode)
    vk_buffer: Any | None = None
    vk_memory: Any | None = None
    # CPU fallback
    cpu_array: np.ndarray | None = None

    def as_numpy(self, dtype: np.dtype = np.float32) -> np.ndarray:
        """Return a writable numpy view of the buffer contents.

        In CPU mode this returns the internal array. In GPU mode the buffer
        is mapped, copied, and unmapped.
        """
        if self.cpu_array is not None:
            return self.cpu_array
        raise RuntimeError("GPU buffer as_numpy not yet implemented")


class BufferManager:
    """Creates, uploads, and patches Vulkan or CPU buffers."""

    def __init__(self, context: VulkanContext) -> None:
        self._ctx = context
        self._buffers: dict[str, ManagedBuffer] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def allocate(
        self,
        name: str,
        size_bytes: int,
        usage: str = "storage",
    ) -> ManagedBuffer:
        """Allocate a new buffer.

        Args:
            name: Logical name (e.g. ``"embeddings"``).
            size_bytes: Size in bytes.
            usage: One of ``"storage"``, ``"uniform"``, ``"staging"``.

        Returns:
            A :class:`ManagedBuffer` handle.
        """
        if self._ctx.cpu_mode:
            # CPU fallback: allocate a flat uint8 array
            arr = np.zeros(size_bytes, dtype=np.uint8)
            buf = ManagedBuffer(
                name=name,
                size_bytes=size_bytes,
                cpu_array=arr,
            )
            self._buffers[name] = buf
            logger.debug("Allocated CPU buffer %s (%d bytes)", name, size_bytes)
            return buf

        # Vulkan path
        buf, mem = self._create_vk_buffer(size_bytes, usage)
        managed = ManagedBuffer(
            name=name,
            size_bytes=size_bytes,
            vk_buffer=buf,
            vk_memory=mem,
        )
        self._buffers[name] = managed
        logger.debug("Allocated GPU buffer %s (%d bytes)", name, size_bytes)
        return managed

    def upload(self, name: str, data: bytes, offset: int = 0) -> None:
        """Write *data* into buffer *name* at *offset*.

        Args:
            name: Buffer logical name.
            data: Raw bytes to write.
            offset: Byte offset into the buffer.
        """
        buf = self._get_buf(name)
        end = offset + len(data)
        if end > buf.size_bytes:
            raise ValueError(
                f"Upload overflow: offset={offset} + len={len(data)} "
                f"> size={buf.size_bytes}"
            )

        if buf.cpu_array is not None:
            buf.cpu_array[offset:end] = np.frombuffer(data, dtype=np.uint8)
            return

        self._upload_vk(buf, data, offset)

    def download(self, name: str, offset: int = 0, size: int | None = None) -> bytes:
        """Read bytes back from buffer *name*.

        Args:
            name: Buffer logical name.
            offset: Byte offset.
            size: Number of bytes to read (default: to end of buffer).

        Returns:
            Raw byte string.
        """
        buf = self._get_buf(name)
        if size is None:
            size = buf.size_bytes - offset
        if buf.cpu_array is not None:
            return bytes(buf.cpu_array[offset : offset + size])
        return self._download_vk(buf, offset, size)

    def patch(
        self,
        name: str,
        data: bytes,
        offset: int = 0,
    ) -> None:
        """In-place patch of a subregion (same semantics as ``upload``)."""
        self.upload(name, data, offset)

    def get_buffer(self, name: str) -> ManagedBuffer:
        return self._get_buf(name)

    def destroy(self, name: str | None = None) -> None:
        """Free one or all buffers."""
        if name is not None:
            buf = self._buffers.pop(name, None)
            if buf and buf.vk_buffer is not None:
                self._destroy_vk_buffer(buf)
            return

        for buf in list(self._buffers.values()):
            if buf.vk_buffer is not None:
                self._destroy_vk_buffer(buf)
        self._buffers.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_buf(self, name: str) -> ManagedBuffer:
        if name not in self._buffers:
            raise KeyError(f"Buffer '{name}' not found")
        return self._buffers[name]

    # ------------------------------------------------------------------
    # Vulkan internals
    # ------------------------------------------------------------------
    def _create_vk_buffer(
        self,
        size_bytes: int,
        usage: str,
    ) -> tuple[Any, Any]:
        assert vk is not None
        dev = self._ctx.vk_device
        assert dev is not None

        usage_bits = {
            "storage": vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            "uniform": vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            "staging": vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        }.get(usage, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)

        buffer_info = vk.VkBufferCreateInfo(
            size=size_bytes,
            usage=usage_bits,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buf = vk.vkCreateBuffer(dev.device, buffer_info, None)

        mem_req = vk.vkGetBufferMemoryRequirements(dev.device, buf)
        mem_type_index = self._find_memory_type(
            mem_req.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        if mem_type_index is None:
            raise RuntimeError("No suitable host-visible memory type found")

        alloc_info = vk.VkMemoryAllocateInfo(
            allocationSize=mem_req.size,
            memoryTypeIndex=mem_type_index,
        )
        mem = vk.vkAllocateMemory(dev.device, alloc_info, None)
        vk.vkBindBufferMemory(dev.device, buf, mem, 0)
        return buf, mem

    def _find_memory_type(self, type_bits: int, properties: int) -> int | None:
        assert vk is not None
        dev = self._ctx.vk_device
        assert dev is not None
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(dev.physical_device)
        for i in range(mem_props.memoryTypeCount):
            if (type_bits & (1 << i)) and (
                mem_props.memoryTypes[i].propertyFlags & properties
            ) == properties:
                return i
        return None

    def _upload_vk(self, buf: ManagedBuffer, data: bytes, offset: int) -> None:
        assert vk is not None
        dev = self._ctx.vk_device
        assert dev is not None
        assert buf.vk_buffer is not None and buf.vk_memory is not None

        mapped = vk.vkMapMemory(dev.device, buf.vk_memory, offset, len(data), 0)
        # cffi memory copy
        src = vk.ffi.new("unsigned char[]", len(data))
        vk.ffi.memmove(src, data, len(data))
        vk.ffi.memmove(mapped, src, len(data))
        vk.vkUnmapMemory(dev.device, buf.vk_memory)

    def _download_vk(self, buf: ManagedBuffer, offset: int, size: int) -> bytes:
        assert vk is not None
        dev = self._ctx.vk_device
        assert dev is not None
        assert buf.vk_buffer is not None and buf.vk_memory is not None

        mapped = vk.vkMapMemory(dev.device, buf.vk_memory, offset, size, 0)
        data = bytes(vk.ffi.buffer(mapped, size))
        vk.vkUnmapMemory(dev.device, buf.vk_memory)
        return data

    def _destroy_vk_buffer(self, buf: ManagedBuffer) -> None:
        assert vk is not None
        dev = self._ctx.vk_device
        assert dev is not None
        if buf.vk_buffer:
            vk.vkDestroyBuffer(dev.device, buf.vk_buffer, None)
        if buf.vk_memory:
            vk.vkFreeMemory(dev.device, buf.vk_memory, None)
        buf.vk_buffer = None
        buf.vk_memory = None
