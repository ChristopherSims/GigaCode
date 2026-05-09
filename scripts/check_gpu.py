#!/usr/bin/env python3
"""Check for GPU availability and recommend installation.

This script detects:
1. CUDA presence and version
2. cuDNN availability
3. Current FAISS installation (CPU vs GPU)
4. Recommendations for optimal setup

Usage:
    python scripts/check_gpu.py
"""

import subprocess
import sys


def check_cuda():
    """Check if CUDA is available and get version."""
    try:
        import torch

        if torch.cuda.is_available():
            return True, torch.version.cuda
        return False, None
    except ImportError:
        return False, None


def check_cudnn():
    """Check if cuDNN is available."""
    try:
        import torch

        if torch.backends.cudnn.enabled:
            return True, torch.backends.cudnn.version()
        return False, None
    except (ImportError, Exception):
        return False, None


def check_faiss_gpu():
    """Check if faiss-gpu is installed."""
    try:
        import faiss

        # faiss-gpu defines this attribute
        has_gpu = hasattr(faiss, "IndexGPU")
        return has_gpu
    except ImportError:
        return False


def get_cuda_capability():
    """Get GPU capability version if available."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
        return None
    except Exception:
        return None


def main():
    print("=" * 60)
    print("GigaCode GPU Configuration Check")
    print("=" * 60)
    print()

    # Check CUDA
    cuda_available, cuda_version = check_cuda()
    print(f"CUDA Available: {'✓ Yes' if cuda_available else '✗ No'}")
    if cuda_version:
        print(f"  CUDA Version: {cuda_version}")
    print()

    # Check cuDNN
    cudnn_available, cudnn_version = check_cudnn()
    print(f"cuDNN Available: {'✓ Yes' if cudnn_available else '✗ No'}")
    if cudnn_version:
        print(f"  cuDNN Version: {cudnn_version}")
    print()

    # Check GPU capability
    capability = get_cuda_capability()
    if capability:
        print(f"GPU Compute Capability: {capability[0]}.{capability[1]}")
        print()

    # Check FAISS
    faiss_gpu = check_faiss_gpu()
    print(f"FAISS GPU Support: {'✓ Installed' if faiss_gpu else '✗ Not installed'}")
    print()

    # Recommendations
    print("=" * 60)
    print("Recommendations")
    print("=" * 60)
    print()

    if cuda_available and cudnn_available and not faiss_gpu:
        print("✓ GPU is available and properly configured!")
        print()
        print("To enable GPU acceleration for FAISS:")
        print("  pip uninstall faiss-cpu")
        print("  pip install 'faiss-gpu~=1.8.0'")
        print()
        print("Or install with GPU support from the start:")
        print("  pip install '.[gpu]'")
        return 0

    elif cuda_available and cudnn_available and faiss_gpu:
        print("✓ GPU acceleration is fully enabled!")
        print()
        print("FAISS will use GPU for fast similarity search.")
        print("Expected: sub-millisecond search on GPU")
        return 0

    elif cuda_available and not cudnn_available:
        print("⚠ CUDA is available but cuDNN is not installed.")
        print()
        print("Install cuDNN for GPU-accelerated operations:")
        print("  https://developer.nvidia.com/cudnn")
        print()
        print("For conda users:")
        print("  conda install -c conda-forge cudnn")
        return 1

    else:
        print("ℹ GPU is not available on this system.")
        print()
        print("GigaCode will use CPU-based FAISS.")
        print("Expected: single-digit millisecond search on CPU")
        print()
        print("To enable GPU support (on a GPU machine):")
        print("  1. Install NVIDIA GPU drivers")
        print("  2. Install CUDA Toolkit")
        print("  3. Install cuDNN")
        print("  4. Run: pip install '.[gpu]'")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
