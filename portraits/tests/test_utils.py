"""Tests for utility functions."""

from pathlib import Path

from portraits.core.utils import (
    ensure_output_dir,
    format_duration,
    get_output_filename,
    get_timestamp,
)


def test_get_timestamp_format():
    """Test timestamp format is correct."""
    ts = get_timestamp()
    assert len(ts) == 15  # YYYYMMDD_HHMMSS
    assert "_" in ts


def test_get_timestamp_custom_format():
    """Test custom timestamp format."""
    ts = get_timestamp("%Y-%m-%d")
    assert len(ts) == 10  # YYYY-MM-DD
    assert ts.count("-") == 2


def test_ensure_output_dir_creates_directory(tmp_path):
    """Test that directory is created."""
    output_dir = tmp_path / "test_output"
    assert not output_dir.exists()

    result = ensure_output_dir(output_dir)

    assert result == output_dir
    assert result.exists()
    assert result.is_dir()


def test_ensure_output_dir_accepts_string(tmp_path):
    """Test that string paths work."""
    output_dir = str(tmp_path / "test_output")
    result = ensure_output_dir(output_dir)

    assert isinstance(result, Path)
    assert result.exists()


def test_get_output_filename_without_index():
    """Test filename generation without index."""
    filename = get_output_filename("test", "png")

    assert filename.startswith("test_")
    assert filename.endswith(".png")
    assert "_" in filename


def test_get_output_filename_with_index():
    """Test filename generation with index."""
    filename = get_output_filename("test", "png", index=5)

    assert filename.startswith("test_")
    assert filename.endswith("_05.png")
    assert "_" in filename


def test_get_output_filename_custom_timestamp():
    """Test filename with custom timestamp."""
    custom_ts = "20250101_120000"
    filename = get_output_filename("test", "png", timestamp=custom_ts)

    assert filename == f"test_{custom_ts}.png"


def test_format_duration_seconds():
    """Test duration formatting for seconds only."""
    assert format_duration(45) == "45s"
    assert format_duration(0) == "0s"


def test_format_duration_minutes():
    """Test duration formatting with minutes."""
    assert format_duration(90) == "1m 30s"
    assert format_duration(120) == "2m"


def test_format_duration_hours():
    """Test duration formatting with hours."""
    assert format_duration(3665) == "1h 1m 5s"
    assert format_duration(7200) == "2h"
