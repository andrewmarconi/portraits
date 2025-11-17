"""Tests for device detection utilities."""

import pytest
import torch

from portraits.core.device import get_device_and_dtype


def test_get_device_and_dtype_returns_valid_types():
    """Test that device detection returns valid types."""
    device, dtype = get_device_and_dtype()

    # Should return valid device string
    assert device in ["cuda", "mps", "cpu"]
    assert isinstance(device, str)

    # Should return valid torch dtype
    assert dtype in [torch.bfloat16, torch.float16, torch.float32]


def test_cuda_uses_bfloat16():
    """Test that CUDA devices use bfloat16."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device, dtype = get_device_and_dtype()
    assert device == "cuda"
    assert dtype == torch.bfloat16


def test_mps_uses_float16():
    """Test that MPS devices use float16."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    device, dtype = get_device_and_dtype()
    assert device == "mps"
    assert dtype == torch.float16


def test_cpu_fallback():
    """Test CPU fallback when no GPU available."""
    # This test will pass on CPU-only systems
    device, dtype = get_device_and_dtype()

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        assert device == "cpu"
        assert dtype == torch.float16
