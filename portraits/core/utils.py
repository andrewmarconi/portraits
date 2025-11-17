"""Shared utility functions."""

import warnings
from datetime import datetime
from pathlib import Path

# Configure warnings globally
warnings.filterwarnings("ignore")


def get_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """Generate timestamp string for file naming.

    Args:
        format: strftime format string (default: YYYYMMDD_HHMMSS)

    Returns:
        str: Timestamp in specified format

    Examples:
        >>> ts = get_timestamp()
        >>> len(ts)
        15
        >>> get_timestamp("%Y-%m-%d")
        '2025-11-17'
    """
    return datetime.now().strftime(format)


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Ensure output directory exists.

    Args:
        output_dir: Directory path as string or Path object

    Returns:
        Path: Resolved output directory path

    Examples:
        >>> output_path = ensure_output_dir("output")
        >>> output_path.exists()
        True
    """
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_output_filename(
    base_name: str, extension: str, index: int | None = None, timestamp: str | None = None
) -> str:
    """Generate output filename with timestamp and optional index.

    Args:
        base_name: Base name (e.g., 'headshot', 'video')
        extension: File extension without dot (e.g., 'png', 'mp4')
        index: Optional index for batch generation
        timestamp: Optional custom timestamp (generated if None)

    Returns:
        str: Filename with timestamp and optional index

    Examples:
        >>> get_output_filename("headshot", "png")
        'headshot_20251117_143022.png'
        >>> get_output_filename("headshot", "png", index=1)
        'headshot_20251117_143022_01.png'
    """
    if timestamp is None:
        timestamp = get_timestamp()

    if index is not None:
        return f"{base_name}_{timestamp}_{index:02d}.{extension}"
    return f"{base_name}_{timestamp}.{extension}"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        str: Formatted duration (e.g., "2m 30s", "45s")

    Examples:
        >>> format_duration(45)
        '45s'
        >>> format_duration(150)
        '2m 30s'
        >>> format_duration(3665)
        '1h 1m 5s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)
