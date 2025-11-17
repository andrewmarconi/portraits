"""Device and dtype detection utilities."""

import os

import torch


def get_device_and_dtype() -> tuple[str, torch.dtype]:
    """Auto-detect optimal device and dtype for the current system.

    Returns:
        tuple: (device, dtype) where device is 'cuda'|'mps'|'cpu'
               and dtype is the optimal torch dtype

    Examples:
        >>> device, dtype = get_device_and_dtype()
        >>> print(f"Using {device} with {dtype}")
    """
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print("✓ Using CUDA GPU with bfloat16")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  Available VRAM: {vram_gb:.1f}GB")

        if vram_gb < 8:
            print("  ⚠ Warning: Low VRAM. Consider enabling CPU offloading.")

    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print("✓ Using Apple Silicon (MPS) with float16")
        print("  Note: MPS uses unified memory architecture")

        # Check PyTorch version for MPS compatibility
        pytorch_version = torch.__version__
        major, minor = pytorch_version.split(".")[:2]
        if int(major) < 2:
            print(f"  ⚠ Warning: PyTorch {pytorch_version} - recommend 2.0+ for MPS")

        # Enable MPS fallback
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    else:
        device = "cpu"
        dtype = torch.float16
        print("⚠ Using CPU with float16 (slow)")
        print("  Recommend: Install CUDA toolkit for GPU acceleration")

    return device, dtype
