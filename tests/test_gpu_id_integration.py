#!/usr/bin/env python3
"""Test GPU ID configuration integration across modules."""

import sys
from pathlib import Path

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent))

from gigacode.gpu_index import GpuIndex


def test_gpu_index_gpu_id_parameter():
    """Test GpuIndex accepts and stores gpu_id parameter."""
    # Test default value
    idx_default = GpuIndex(384, use_gpu=False)
    assert idx_default.gpu_id == 0, "Default gpu_id should be 0"
    print("GpuIndex default gpu_id=0")
    
    # Test custom value
    idx_custom = GpuIndex(384, use_gpu=False, gpu_id=3)
    assert idx_custom.gpu_id == 3, "gpu_id should be settable"
    print("GpuIndex custom gpu_id=3")
    
    # Test with GPU enabled (but no actual GPU available)
    idx_gpu = GpuIndex(384, use_gpu=True, gpu_id=1)
    assert idx_gpu.gpu_id == 1, "gpu_id should persist with GPU enabled"
    print("GpuIndex gpu_id=1 with use_gpu=True")


def test_backward_compatibility():
    """Test that code without gpu_id parameter still works."""
    # This simulates old code that doesn't pass gpu_id
    idx = GpuIndex(dim=768, use_gpu=False)
    assert hasattr(idx, 'gpu_id'), "GpuIndex should have gpu_id attribute"
    assert idx.gpu_id == 0, "Default gpu_id should be 0 for backward compatibility"
    print("Backward compatibility: gpu_id parameter optional")


def test_embedding_dim_with_gpu_id():
    """Test various embedding dimensions work with gpu_id."""
    for dim in [256, 384, 512, 768, 1024]:
        idx = GpuIndex(dim, use_gpu=False, gpu_id=0)
        assert idx.dim == dim, f"Dimension should be {dim}"
        assert idx.gpu_id == 0, "gpu_id should be 0"
    print(f"All embedding dimensions work with gpu_id parameter")


if __name__ == "__main__":
    print("Testing GPU ID Configuration Integration\n" + "=" * 40)
    
    test_gpu_index_gpu_id_parameter()
    test_backward_compatibility()
    test_embedding_dim_with_gpu_id()
    
    print("\n" + "=" * 40)
    print("All GPU ID integration tests PASSED!")
