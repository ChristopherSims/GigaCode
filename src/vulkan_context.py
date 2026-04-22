"""Vulkan compute context: instance, device, queue, and command management.

The module tries to use the ``vulkan`` Python package (cvulkan). If that is
not available it falls back to a CPU-only stub so that the rest of the
pipeline can still be exercised.
"""

from __future__ import annotations

import logging
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional vulkan bindings
# ---------------------------------------------------------------------------
try:
    import vulkan as vk  # type: ignore[import-untyped]

    _HAS_VULKAN = True
except Exception as exc:
    logger.warning("vulkan Python package not available (%s). Using CPU fallback.", exc)
    _HAS_VULKAN = False
    vk = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# CPU fallback helpers
# ---------------------------------------------------------------------------
def _cpu_dot_search(
    embeddings: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Brute-force dot-product search on the CPU.

    Args:
        embeddings: Array of shape ``(N, D)``.
        query: Array of shape ``(D,)``.

    Returns:
        Array of shape ``(N,)`` with similarity scores.
    """
    return np.dot(embeddings, query)


def _cpu_cluster_regions(
    embeddings: np.ndarray,
    threshold: float,
    window: int = 16,
) -> list[dict[str, Any]]:
    """Greedy clustering on the CPU.

    Args:
        embeddings: Array of shape ``(N, D)``.
        threshold: Similarity threshold.
        window: Max scan-ahead distance.

    Returns:
        List of cluster dicts with ``start_token``, ``end_token``, ``count``,
        ``avg_score``.
    """
    n = embeddings.shape[0]
    clusters: list[dict[str, Any]] = []
    i = 0
    while i < n:
        start = i
        end = i
        count = 1
        score_sum = 0.0
        for offset in range(1, window + 1):
            j = i + offset
            if j >= n:
                break
            s = float(np.dot(embeddings[j - 1], embeddings[j]))
            if s > threshold:
                end = j
                count += 1
                score_sum += s
            else:
                break
        clusters.append(
            {
                "start_token": int(start),
                "end_token": int(end),
                "count": int(count),
                "avg_score": float(score_sum / max(1, count - 1)),
            }
        )
        i = end + 1
    return clusters


# ---------------------------------------------------------------------------
# Vulkan context
# ---------------------------------------------------------------------------
@dataclass
class VulkanDevice:
    """Minimal Vulkan device handle bundle."""

    instance: Any
    physical_device: Any
    device: Any
    queue: Any
    queue_family_index: int
    command_pool: Any
    properties: Any


class VulkanContext:
    """Manages Vulkan instance, device, and compute queue.

    If the ``vulkan`` package is unavailable the context switches to CPU
    fallback mode.
    """

    def __init__(self) -> None:
        self._vk_device: VulkanDevice | None = None
        self._cpu_mode = not _HAS_VULKAN
        if not self._cpu_mode:
            try:
                self._vk_device = self._create_vulkan_device()
                logger.info("Vulkan compute device ready.")
            except Exception as exc:
                logger.warning("Vulkan init failed (%s). Falling back to CPU.", exc)
                self._cpu_mode = True
                self._vk_device = None

    @property
    def cpu_mode(self) -> bool:
        """True when running without a Vulkan GPU."""
        return self._cpu_mode

    @property
    def vk_device(self) -> VulkanDevice | None:
        return self._vk_device

    # ------------------------------------------------------------------
    # Vulkan init
    # ------------------------------------------------------------------
    def _create_vulkan_device(self) -> VulkanDevice:
        assert vk is not None

        app_info = vk.VkApplicationInfo(
            pApplicationName="GigaCode",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="GigaCode",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )

        layers = []
        if os.environ.get("VK_LAYER_PATH") or os.environ.get("ENABLE_VULKAN_VALIDATION"):
            layers.append("VK_LAYER_KHRONOS_validation")

        create_info = vk.VkInstanceCreateInfo(
            pApplicationInfo=app_info,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers if layers else None,
            enabledExtensionCount=0,
            ppEnabledExtensionNames=None,
        )

        instance = vk.vkCreateInstance(create_info, None)

        physical_devices = vk.vkEnumeratePhysicalDevices(instance)
        if not physical_devices:
            raise RuntimeError("No Vulkan physical devices found")

        # Prefer discrete GPU, then integrated, then any
        physical_device = physical_devices[0]
        for pdev in physical_devices:
            props = vk.vkGetPhysicalDeviceProperties(pdev)
            if props.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                physical_device = pdev
                break

        props = vk.vkGetPhysicalDeviceProperties(physical_device)
        logger.info("Selected GPU: %s", props.deviceName)

        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
        compute_qfi = None
        for i, qf in enumerate(queue_families):
            if qf.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                compute_qfi = i
                break
        if compute_qfi is None:
            raise RuntimeError("No compute queue family found")

        queue_priority = [1.0]
        device_queue_info = vk.VkDeviceQueueCreateInfo(
            queueFamilyIndex=compute_qfi,
            queueCount=1,
            pQueuePriorities=queue_priority,
        )

        device_features = vk.VkPhysicalDeviceFeatures()
        device_create_info = vk.VkDeviceCreateInfo(
            queueCreateInfoCount=1,
            pQueueCreateInfos=[device_queue_info],
            enabledExtensionCount=0,
            ppEnabledExtensionNames=None,
            pEnabledFeatures=[device_features],
        )

        device = vk.vkCreateDevice(physical_device, device_create_info, None)
        queue = vk.vkGetDeviceQueue(device, compute_qfi, 0)

        command_pool = vk.vkCreateCommandPool(
            device,
            vk.VkCommandPoolCreateInfo(
                queueFamilyIndex=compute_qfi,
                flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            ),
            None,
        )

        return VulkanDevice(
            instance=instance,
            physical_device=physical_device,
            device=device,
            queue=queue,
            queue_family_index=compute_qfi,
            command_pool=command_pool,
            properties=props,
        )

    def destroy(self) -> None:
        """Clean up Vulkan objects."""
        if self._vk_device is None:
            return
        d = self._vk_device
        assert vk is not None
        vk.vkDestroyCommandPool(d.device, d.command_pool, None)
        vk.vkDestroyDevice(d.device, None)
        vk.vkDestroyInstance(d.instance, None)
        self._vk_device = None

    def __enter__(self) -> VulkanContext:
        return self

    def __exit__(self, *args: Any) -> None:
        self.destroy()

    # ------------------------------------------------------------------
    # Compute helpers
    # ------------------------------------------------------------------
    def similarity_search(
        self,
        embeddings: np.ndarray,
        query: np.ndarray,
    ) -> np.ndarray:
        """Return similarity scores for each token against *query*.

        Args:
            embeddings: ``(N, D)`` float32 array.
            query: ``(D,)`` float32 array.

        Returns:
            ``(N,)`` float32 scores.
        """
        if self._cpu_mode:
            return _cpu_dot_search(embeddings, query)
        # TODO: GPU compute shader dispatch
        logger.warning("GPU similarity_search not yet implemented; using CPU fallback.")
        return _cpu_dot_search(embeddings, query)

    def cluster_regions(
        self,
        embeddings: np.ndarray,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Cluster similar token regions.

        Args:
            embeddings: ``(N, D)`` float32 array.
            threshold: Similarity threshold.

        Returns:
            List of cluster metadata dicts.
        """
        if self._cpu_mode:
            return _cpu_cluster_regions(embeddings, threshold)
        # TODO: GPU compute shader dispatch
        logger.warning("GPU cluster_regions not yet implemented; using CPU fallback.")
        return _cpu_cluster_regions(embeddings, threshold)
