# Portraits Codebase Cleanup Plan

**Status:** Ready for Implementation
**Created:** 2025-11-17
**Estimated Total Time:** 6-8 hours

---

## Executive Summary

This plan addresses critical code duplication, structural inefficiencies, and maintenance issues in the Portraits multimodal AI generation project. The codebase currently has ~2,000 lines across 6 files with significant duplication and no shared infrastructure.

**Key Issues:**
- Duplicate device detection logic (3 implementations, 60+ duplicate lines)
- No centralized configuration (hardcoded paths scattered across files)
- Flat file structure (no module hierarchy)
- Unused entry point (main.py is placeholder)
- No optional dependencies (all-or-nothing installation)

**Expected Benefits:**
- 60% reduction in code duplication (~150 lines eliminated)
- Faster installation (5 vs 15 packages for image-only)
- Easier maintenance (centralized config)
- Professional package structure
- Better testability

---

## Phase 1: Critical Fixes (1-2 hours)

### 1.1 Create Unified Device Detection Module

**File:** `portraits/core/device.py`

**Problem:** Device detection logic duplicated in 3 places:
- `generate_video.py:56-84` - `get_device_and_dtype()` function (29 lines)
- `generate_voice.py:57-80` - Identical function (24 lines)
- `generate_images.py:38-58` - Inline implementation (21 lines)

**Implementation:**

```python
# portraits/core/device.py
"""Device and dtype detection utilities."""
import torch
from typing import Tuple

def get_device_and_dtype() -> Tuple[str, torch.dtype]:
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
        print(f"✓ Using CUDA GPU with bfloat16")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  Available VRAM: {vram_gb:.1f}GB")

        if vram_gb < 8:
            print(f"  ⚠ Warning: Low VRAM. Consider enabling CPU offloading.")

    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print(f"✓ Using Apple Silicon (MPS) with float16")
        print(f"  Note: MPS uses unified memory architecture")

        # Check PyTorch version for MPS compatibility
        import torch
        pytorch_version = torch.__version__
        major, minor = pytorch_version.split('.')[:2]
        if int(major) < 2:
            print(f"  ⚠ Warning: PyTorch {pytorch_version} - recommend 2.0+ for MPS")

        # Enable MPS fallback
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    else:
        device = "cpu"
        dtype = torch.float16
        print(f"⚠ Using CPU with float16 (slow)")
        print(f"  Recommend: Install CUDA toolkit for GPU acceleration")

    return device, dtype
```

**Files to Update:**
1. `generate_video.py` - Replace lines 56-84 with `from core.device import get_device_and_dtype`
2. `generate_voice.py` - Replace lines 57-80 with import
3. `generate_images.py` - Replace inline detection (lines 38-58) with function call

**Testing:**
```bash
# Test import
uv run python -c "from core.device import get_device_and_dtype; print(get_device_and_dtype())"
```

---

### 1.2 Create Shared Utilities Module

**File:** `portraits/core/utils.py`

**Problem:** Repeated patterns across files:
- Warning suppression: 4 identical lines
- Timestamp generation: 3 identical implementations
- Output directory creation: 3 different approaches

**Implementation:**

```python
# portraits/core/utils.py
"""Shared utility functions."""
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

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

def ensure_output_dir(output_dir: Union[str, Path]) -> Path:
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
    base_name: str,
    extension: str,
    index: Optional[int] = None,
    timestamp: Optional[str] = None
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
```

**Files to Update:**
1. `generate_images.py` - Replace timestamp/directory code
2. `generate_video.py` - Replace timestamp/directory code
3. `generate_voice.py` - Replace timestamp/directory code
4. Remove `warnings.filterwarnings("ignore")` from all files

---

### 1.3 Create Configuration Management System

**File:** `portraits/config.yaml`

**Problem:** Hardcoded values scattered across files:
- Model paths repeated 8+ times
- Output directory repeated 4 times
- Default parameters mixed throughout

**Implementation:**

```yaml
# portraits/config.yaml
# Portraits Configuration File
# Edit these values to customize default behavior

# Model paths
models:
  # Image generation (SDXL Turbo)
  sdxl_turbo: "stabilityai/sdxl-turbo"

  # Text-to-video generation (SkyReels-V2)
  skyreels_v2: "./SkyReels-V2-T2V-14B-540P"

  # Text-to-speech (Maya1)
  maya1: "maya-research/maya1"
  maya1_tokenizer: "maya-research/maya1"

  # SNAC neural codec for voice
  snac: "hubertsiuzdak/snac_24khz"

# File paths
paths:
  # Output directory for all generated content
  output_dir: "output"

  # Hugging Face cache location (uses HF_HOME env var if set)
  hf_cache: "${HF_HOME:~/.cache/huggingface}"

  # Default LoRA path (can be overridden)
  default_lora: null

# Default parameters for image generation
image_generation:
  width: 512
  height: 512
  num_inference_steps: 2  # SDXL Turbo optimized
  guidance_scale: -100     # Disabled for Turbo (-100)
  clip_skip: 1
  num_images: 1
  lora_scale: 1.0
  variant: "fp16"

# Default parameters for video generation
video_generation:
  # Frame settings
  num_frames: 97           # ~4 seconds at 24fps
  fps: 24

  # Resolution (540P)
  width: 960
  height: 544

  # Generation quality
  guidance_scale: 6.0
  num_inference_steps: 50

  # Memory optimization
  enable_offload: true
  enable_attention_slicing: true
  enable_vae_slicing: true

# Default parameters for voice generation
voice_generation:
  # Sampling parameters
  temperature: 0.4         # Lower = more consistent
  top_p: 0.9              # Nucleus sampling
  max_tokens: 2048        # ~20-30 seconds of audio

  # Audio settings
  sample_rate: 24000      # 24kHz mono

  # Default voice description
  default_voice: "medium pitch, warm, conversational"

# Default parameters for mesh morphing
mesh_morphing:
  fps: 24
  morph_frames: 8          # Frames between each image
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
  max_num_faces: 1
  refine_landmarks: true   # Include iris landmarks
  visualize_landmarks: false

# Logging and debugging
logging:
  level: "INFO"           # DEBUG, INFO, WARNING, ERROR
  show_warnings: false
  show_progress: true
```

**File:** `portraits/core/config.py`

```python
# portraits/core/config.py
"""Configuration management for Portraits."""
import os
import yaml
from pathlib import Path
from typing import Any, Optional

class Config:
    """Centralized configuration manager.

    Loads configuration from config.yaml and provides convenient access
    to settings with environment variable expansion and defaults.

    Examples:
        >>> config = Config()
        >>> model_id = config.get('models.sdxl_turbo')
        >>> output_dir = config.get('paths.output_dir')
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_path: Path to config file (default: ./config.yaml)
        """
        if config_path is None:
            # Look for config.yaml in project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Please create config.yaml in the project root."
            )

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        return self._expand_env_vars(config)

    def _expand_env_vars(self, obj: Any) -> Any:
        """Recursively expand environment variables in config values.

        Supports syntax: ${VAR_NAME:default_value}
        """
        if isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Expand environment variables
            if obj.startswith('${') and '}' in obj:
                var_expr = obj[2:obj.index('}')]
                if ':' in var_expr:
                    var_name, default = var_expr.split(':', 1)
                    return os.environ.get(var_name, default)
                else:
                    return os.environ.get(var_expr, obj)
            return obj
        else:
            return obj

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path (e.g., 'models.sdxl_turbo')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config.get('models.sdxl_turbo')
            'stabilityai/sdxl-turbo'
            >>> config.get('image_generation.width')
            512
            >>> config.get('nonexistent.key', 'fallback')
            'fallback'
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value (runtime only, not persisted).

        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = self._load_config()

# Global config instance (singleton pattern)
_config_instance: Optional[Config] = None

def get_config() -> Config:
    """Get global configuration instance.

    Returns:
        Config: Global configuration object
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

# Convenience alias
config = get_config()
```

**Files to Update:**
1. All generator files - replace hardcoded model paths
2. All generator files - use config for default parameters

**Testing:**
```bash
# Test config loading
uv run python -c "from core.config import config; print(config.get('models.sdxl_turbo'))"
```

---

### 1.4 Fix main.py CLI Entry Point

**File:** `portraits/main.py`

**Problem:** Current `main.py` is a 6-line placeholder that does nothing.

**Implementation:**

```python
#!/usr/bin/env python3
"""Unified CLI entry point for Portraits multimodal generation."""
import argparse
import sys
from pathlib import Path

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='portraits',
        description='Portraits: Multimodal AI Generation Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate images
  %(prog)s image --prompt "professional headshot of a woman"
  %(prog)s image --prompt "portrait" --num-images 10 --lora-path ./my_lora.safetensors

  # Generate videos
  %(prog)s video --prompt "cinematic drone shot over mountains"
  %(prog)s video --prompt "underwater coral reef" --num-frames 145

  # Generate voice
  %(prog)s voice --text "Hello world!" --voice "warm, friendly, medium pitch"
  %(prog)s voice --text "Welcome to the show! <excited>" --voice "energetic, high pitch"

  # Create morphing video
  %(prog)s morph --input output/ --output morphed.mp4
  %(prog)s morph --input images/ --output result.mp4 --fps 30 --morph-frames 12

  # Launch web UI
  %(prog)s ui
  %(prog)s ui --share  # Create public link
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Generation mode')

    # ========== Image Generation ==========
    img_parser = subparsers.add_parser(
        'image',
        help='Generate images using SDXL Turbo',
        description='Generate professional images using SDXL Turbo with optional LoRA weights'
    )
    img_parser.add_argument(
        '--prompt',
        required=True,
        help='Text prompt for image generation'
    )
    img_parser.add_argument(
        '--num-images',
        type=int,
        default=1,
        help='Number of images to generate (default: 1)'
    )
    img_parser.add_argument(
        '--width',
        type=int,
        default=512,
        help='Image width in pixels (default: 512)'
    )
    img_parser.add_argument(
        '--height',
        type=int,
        default=512,
        help='Image height in pixels (default: 512)'
    )
    img_parser.add_argument(
        '--lora-path',
        help='Path to LoRA weights (.safetensors file or HuggingFace repo)'
    )
    img_parser.add_argument(
        '--lora-scale',
        type=float,
        default=1.0,
        help='LoRA adapter strength 0.0-1.0 (default: 1.0)'
    )
    img_parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory (default: output/)'
    )

    # ========== Video Generation ==========
    vid_parser = subparsers.add_parser(
        'video',
        help='Generate videos from text using SkyReels-V2',
        description='Generate videos from text descriptions using SkyReels-V2-T2V-14B-540P'
    )
    vid_parser.add_argument(
        '--prompt',
        required=True,
        help='Text description of the video to generate'
    )
    vid_parser.add_argument(
        '--num-frames',
        type=int,
        default=97,
        help='Number of frames (default: 97 for ~4s at 24fps)'
    )
    vid_parser.add_argument(
        '--fps',
        type=int,
        default=24,
        help='Frames per second (default: 24)'
    )
    vid_parser.add_argument(
        '--guidance-scale',
        type=float,
        default=6.0,
        help='CFG scale for prompt adherence (default: 6.0)'
    )
    vid_parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory (default: output/)'
    )

    # ========== Voice Generation ==========
    voice_parser = subparsers.add_parser(
        'voice',
        help='Generate speech from text using Maya1',
        description='Generate natural speech from text with customizable voice characteristics'
    )
    voice_parser.add_argument(
        '--text',
        required=True,
        help='Text to convert to speech (supports emotion tags like <laugh>, <excited>)'
    )
    voice_parser.add_argument(
        '--voice',
        help='Voice description (e.g., "warm, friendly, medium pitch")'
    )
    voice_parser.add_argument(
        '--temperature',
        type=float,
        default=0.4,
        help='Sampling temperature for variation (default: 0.4)'
    )
    voice_parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory (default: output/)'
    )

    # ========== Mesh Morphing ==========
    morph_parser = subparsers.add_parser(
        'morph',
        help='Create facial morphing videos',
        description='Create smooth morphing videos from a sequence of facial images'
    )
    morph_parser.add_argument(
        '--input',
        required=True,
        help='Input directory containing images'
    )
    morph_parser.add_argument(
        '--output',
        required=True,
        help='Output video file path (e.g., morphed.mp4)'
    )
    morph_parser.add_argument(
        '--fps',
        type=int,
        default=24,
        help='Frames per second (default: 24)'
    )
    morph_parser.add_argument(
        '--morph-frames',
        type=int,
        default=8,
        help='Number of interpolation frames between images (default: 8)'
    )
    morph_parser.add_argument(
        '--visualize',
        action='store_true',
        help='Draw facial landmarks on output (for debugging)'
    )

    # ========== Web UI ==========
    ui_parser = subparsers.add_parser(
        'ui',
        help='Launch Gradio web interface',
        description='Launch interactive web interface for all generation features'
    )
    ui_parser.add_argument(
        '--share',
        action='store_true',
        help='Create public share link (requires internet)'
    )
    ui_parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Server port (default: 7860)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command specified
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Execute command
    try:
        if args.command == 'image':
            from generators.image import generate_headshot
            result = generate_headshot(
                prompt=args.prompt,
                num_images=args.num_images,
                width=args.width,
                height=args.height,
                lora_path=args.lora_path,
                lora_scale=args.lora_scale,
                output_dir=args.output_dir
            )
            print(f"\n✓ Generated {args.num_images} image(s)")

        elif args.command == 'video':
            from generators.video import generate_video
            result = generate_video(
                prompt=args.prompt,
                num_frames=args.num_frames,
                fps=args.fps,
                guidance_scale=args.guidance_scale,
                output_dir=args.output_dir
            )
            print(f"\n✓ Generated video: {result}")

        elif args.command == 'voice':
            from generators.voice import generate_voice
            result = generate_voice(
                text=args.text,
                description=args.voice,
                temperature=args.temperature,
                output_dir=args.output_dir
            )
            print(f"\n✓ Generated audio: {result}")

        elif args.command == 'morph':
            from generators.morph import create_mesh_morphing_video
            result = create_mesh_morphing_video(
                input_dir=args.input,
                output_file=args.output,
                fps=args.fps,
                morph_frames=args.morph_frames,
                visualize_landmarks=args.visualize
            )
            print(f"\n✓ Generated morphing video: {result}")

        elif args.command == 'ui':
            from ui.app import create_app
            print("Launching Gradio web interface...")
            app = create_app()
            app.launch(
                share=args.share,
                server_port=args.port,
                show_error=True
            )

    except ImportError as e:
        print(f"\n❌ Error: Missing dependencies for '{args.command}' command")
        print(f"   {e}")
        print(f"\nInstall with: uv sync --extra {args.command}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Testing:**
```bash
# Test CLI
uv run python -m portraits --help
uv run python -m portraits image --help
```

**Update pyproject.toml:**
```toml
[project.scripts]
portraits = "main:main"
```

---

## Phase 2: Structural Reorganization (2-3 hours)

### 2.1 Reorganize File Structure

**Goal:** Transform flat structure into proper Python package.

**Current Structure:**
```
Portraits/
├── generate_images.py
├── generate_video.py
├── generate_voice.py
├── generate_morph_video.py
├── app.py
├── main.py
└── pyproject.toml
```

**Target Structure:**
```
portraits/
├── core/
│   ├── __init__.py
│   ├── device.py          # From Phase 1.1
│   ├── config.py          # From Phase 1.3
│   ├── utils.py           # From Phase 1.2
│   └── exceptions.py      # New: Custom exceptions
├── generators/
│   ├── __init__.py
│   ├── image.py           # Renamed from generate_images.py
│   ├── video.py           # Renamed from generate_video.py
│   ├── voice.py           # Renamed from generate_voice.py
│   └── morph.py           # Renamed from generate_morph_video.py
├── ui/
│   ├── __init__.py
│   └── app.py             # Moved from root
├── tests/                 # New: Unit tests
│   ├── __init__.py
│   ├── test_device.py
│   ├── test_utils.py
│   ├── test_config.py
│   └── test_generators.py
├── config.yaml            # From Phase 1.3
├── main.py                # From Phase 1.4
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

**Migration Commands:**

```bash
# Create directory structure
mkdir -p portraits/core
mkdir -p portraits/generators
mkdir -p portraits/ui
mkdir -p portraits/tests

# Create __init__.py files
touch portraits/__init__.py
touch portraits/core/__init__.py
touch portraits/generators/__init__.py
touch portraits/ui/__init__.py
touch portraits/tests/__init__.py

# Move files (use git mv to preserve history)
git mv generate_images.py portraits/generators/image.py
git mv generate_video.py portraits/generators/video.py
git mv generate_voice.py portraits/generators/voice.py
git mv generate_morph_video.py portraits/generators/morph.py
git mv app.py portraits/ui/app.py

# Move Phase 1 files
mv core/device.py portraits/core/
mv core/utils.py portraits/core/
mv core/config.py portraits/core/
mv config.yaml portraits/

# Update main.py (already done in Phase 1.4)
```

**Create __init__.py files:**

```python
# portraits/__init__.py
"""Portraits: Multimodal AI Generation Suite."""

__version__ = "0.1.0"

# portraits/core/__init__.py
"""Core utilities for Portraits."""

from .config import config, get_config
from .device import get_device_and_dtype
from .utils import (
    ensure_output_dir,
    get_output_filename,
    get_timestamp,
    format_duration,
)

__all__ = [
    'config',
    'get_config',
    'get_device_and_dtype',
    'ensure_output_dir',
    'get_output_filename',
    'get_timestamp',
    'format_duration',
]

# portraits/generators/__init__.py
"""Generation modules for Portraits."""

from .image import generate_headshot
from .video import generate_video
from .voice import generate_voice
from .morph import create_mesh_morphing_video

__all__ = [
    'generate_headshot',
    'generate_video',
    'generate_voice',
    'create_mesh_morphing_video',
]

# portraits/ui/__init__.py
"""User interface modules."""

from .app import create_app

__all__ = ['create_app']
```

---

### 2.2 Update Import Statements

**Files to Update:**

1. **portraits/generators/image.py** (formerly generate_images.py)

```python
# OLD imports
import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.utils import logging
import os
from datetime import datetime

# NEW imports
import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.utils import logging
from pathlib import Path

# Add local imports
from core.config import config
from core.device import get_device_and_dtype
from core.utils import ensure_output_dir, get_output_filename
```

2. **portraits/generators/video.py** (formerly generate_video.py)

```python
# Remove duplicate get_device_and_dtype() function (lines 56-84)
# Add imports
from core.device import get_device_and_dtype
from core.config import config
from core.utils import ensure_output_dir, get_output_filename
```

3. **portraits/generators/voice.py** (formerly generate_voice.py)

```python
# Remove duplicate get_device_and_dtype() function (lines 57-80)
# Add imports
from core.device import get_device_and_dtype
from core.config import config
from core.utils import ensure_output_dir, get_output_filename
```

4. **portraits/generators/morph.py** (formerly generate_morph_video.py)

```python
# Add imports
from core.config import config
from core.utils import ensure_output_dir
```

5. **portraits/ui/app.py**

```python
# Update imports to use new structure
from generators.image import generate_headshot
from generators.video import generate_video
from generators.voice import generate_voice
from generators.morph import create_mesh_morphing_video
```

---

### 2.3 Split Optional Dependencies

**File:** `pyproject.toml`

**Current:**
```toml
dependencies = [
    "torch>=2.0.0",
    "diffusers>=0.24.0",
    # ... all 15 packages required
]
```

**Updated:**

```toml
[project]
name = "portraits"
version = "0.1.0"
description = "Multimodal AI generation suite: images, videos, speech, and morphing"
readme = "README.md"
requires-python = ">=3.11"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

# Minimal core dependencies
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "Pillow>=9.0.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
# Image generation with SDXL Turbo
image = [
    "diffusers>=0.24.0",
    "transformers>=4.30.0",
    "accelerate>=0.20.0",
    "peft>=0.18.0",
]

# Text-to-video generation with SkyReels-V2
video = [
    "diffusers>=0.24.0",
    "transformers>=4.30.0",
    "accelerate>=0.20.0",
]

# Text-to-speech with Maya1
voice = [
    "transformers>=4.30.0",
    "snac>=1.0.0",
    "soundfile>=0.12.0",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20.0",
    "ftfy>=6.0.0",
]

# Video morphing with MediaPipe
morph = [
    "opencv-python>=4.12.0.88",
    "scipy>=1.16.3",
    "mediapipe>=0.10.14",
]

# Gradio web interface
ui = [
    "gradio>=4.0.0",
]

# Development dependencies
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

# Install all features
all = [
    "portraits[image,video,voice,morph,ui]",
]

[project.scripts]
portraits = "main:main"

[project.urls]
Homepage = "https://github.com/yourusername/portraits"
Documentation = "https://github.com/yourusername/portraits/blob/main/CLAUDE.md"
Repository = "https://github.com/yourusername/portraits"

[tool.uv.workspace]
members = [
    "SkyReels-V2",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# Code quality tools
[tool.black]
line-length = 100
target-version = ['py311']

[tool.ruff]
line-length = 100
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by black)
]

[tool.ruff.isort]
known-first-party = ["core", "generators", "ui"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--cov=portraits --cov-report=html --cov-report=term"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
```

**Installation Examples:**

```bash
# Image generation only (5 packages)
uv sync --extra image

# Video morphing only (3 packages)
uv sync --extra morph

# Image + Voice (10 packages)
uv sync --extra image --extra voice

# Everything (all features)
uv sync --extra all

# Development (includes all + dev tools)
uv sync --extra all --extra dev
```

---

### 2.4 Standardize on pathlib.Path

**Problem:** Mixed use of `os.path` and `pathlib.Path`

**Files to Update:**

**portraits/generators/image.py:**
```python
# OLD (lines 32, 106-107)
os.makedirs(output_dir, exist_ok=True)
filepath = os.path.join(output_dir, filename)

# NEW
from pathlib import Path
output_path = ensure_output_dir(output_dir)  # Returns Path object
filename = get_output_filename("headshot", "png", index=i)
filepath = output_path / filename
```

**All generator files:**
- Replace `os.path.join()` with `/` operator
- Replace `os.makedirs()` with `Path.mkdir()`
- Use `Path.exists()`, `Path.is_file()`, etc.

---

## Phase 3: Code Quality & Polish (1-2 hours)

### 3.1 Add Custom Exceptions

**File:** `portraits/core/exceptions.py`

```python
# portraits/core/exceptions.py
"""Custom exceptions for Portraits."""

class PortraitsError(Exception):
    """Base exception for all Portraits errors."""
    pass

class ConfigurationError(PortraitsError):
    """Configuration-related errors."""
    pass

class ModelNotFoundError(PortraitsError):
    """Model weights not found at specified path."""

    def __init__(self, model_path: str, instructions: str = ""):
        self.model_path = model_path
        message = f"Model not found at '{model_path}'."
        if instructions:
            message += f"\n\n{instructions}"
        super().__init__(message)

class DeviceError(PortraitsError):
    """Device-related errors (CUDA, MPS, CPU)."""
    pass

class GenerationError(PortraitsError):
    """Error during generation process."""
    pass

class FaceDetectionError(PortraitsError):
    """Face detection failed in morphing."""
    pass

def handle_model_load_error(error: Exception, model_path: str) -> None:
    """Handle model loading errors with helpful messages.

    Args:
        error: Original exception
        model_path: Path where model was expected

    Raises:
        ModelNotFoundError: With helpful installation instructions
        GenerationError: For other errors
    """
    error_msg = str(error)

    if "404" in error_msg or "Entry Not Found" in error_msg:
        instructions = (
            "Please download the model manually:\n"
            f"  git lfs install\n"
            f"  git clone https://huggingface.co/{model_path}"
        )
        raise ModelNotFoundError(model_path, instructions) from error

    raise GenerationError(f"Failed to load model: {error}") from error
```

**Usage in generator files:**

```python
from core.exceptions import ModelNotFoundError, handle_model_load_error

try:
    pipe = AutoPipelineForText2Image.from_pretrained(model_id)
except Exception as e:
    handle_model_load_error(e, model_id)
```

---

### 3.2 Add Unit Tests

**File:** `portraits/tests/test_device.py`

```python
# portraits/tests/test_device.py
"""Tests for device detection utilities."""
import pytest
import torch
from core.device import get_device_and_dtype

def test_get_device_and_dtype_returns_valid_types():
    """Test that device detection returns valid types."""
    device, dtype = get_device_and_dtype()

    # Should return valid device string
    assert device in ['cuda', 'mps', 'cpu']
    assert isinstance(device, str)

    # Should return valid torch dtype
    assert dtype in [torch.bfloat16, torch.float16, torch.float32]

def test_cuda_uses_bfloat16():
    """Test that CUDA devices use bfloat16."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device, dtype = get_device_and_dtype()
    assert device == 'cuda'
    assert dtype == torch.bfloat16

def test_mps_uses_float16():
    """Test that MPS devices use float16."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    device, dtype = get_device_and_dtype()
    assert device == 'mps'
    assert dtype == torch.float16

def test_cpu_fallback():
    """Test CPU fallback when no GPU available."""
    # This test will pass on CPU-only systems
    device, dtype = get_device_and_dtype()

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        assert device == 'cpu'
        assert dtype == torch.float16
```

**File:** `portraits/tests/test_utils.py`

```python
# portraits/tests/test_utils.py
"""Tests for utility functions."""
import pytest
from pathlib import Path
from core.utils import (
    get_timestamp,
    ensure_output_dir,
    get_output_filename,
    format_duration,
)

def test_get_timestamp_format():
    """Test timestamp format is correct."""
    ts = get_timestamp()
    assert len(ts) == 15  # YYYYMMDD_HHMMSS
    assert '_' in ts
    assert ts.count('_') == 1

def test_get_timestamp_custom_format():
    """Test custom timestamp format."""
    ts = get_timestamp("%Y-%m-%d")
    assert len(ts) == 10  # YYYY-MM-DD
    assert ts.count('-') == 2

def test_ensure_output_dir_creates_directory(tmp_path):
    """Test that directory is created."""
    output_dir = tmp_path / "output"
    assert not output_dir.exists()

    result = ensure_output_dir(output_dir)

    assert result.exists()
    assert result.is_dir()
    assert result == output_dir

def test_ensure_output_dir_accepts_string(tmp_path):
    """Test that string paths work."""
    output_dir = str(tmp_path / "output")
    result = ensure_output_dir(output_dir)

    assert isinstance(result, Path)
    assert result.exists()

def test_ensure_output_dir_idempotent(tmp_path):
    """Test that calling twice doesn't error."""
    output_dir = tmp_path / "output"

    result1 = ensure_output_dir(output_dir)
    result2 = ensure_output_dir(output_dir)

    assert result1 == result2

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
    assert "_05.png" in filename

def test_get_output_filename_custom_timestamp():
    """Test filename with custom timestamp."""
    filename = get_output_filename("test", "mp4", timestamp="20250101_120000")

    assert filename == "test_20250101_120000.mp4"

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
```

**File:** `portraits/tests/test_config.py`

```python
# portraits/tests/test_config.py
"""Tests for configuration management."""
import pytest
from pathlib import Path
from core.config import Config

@pytest.fixture
def sample_config(tmp_path):
    """Create a sample config file for testing."""
    config_content = """
models:
  test_model: "test/model"

paths:
  output_dir: "output"

defaults:
  width: 512
  height: 512
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return Config(str(config_file))

def test_config_get_simple_key(sample_config):
    """Test getting a simple configuration value."""
    value = sample_config.get('models.test_model')
    assert value == "test/model"

def test_config_get_nested_key(sample_config):
    """Test getting nested configuration values."""
    width = sample_config.get('defaults.width')
    assert width == 512

def test_config_get_missing_key_returns_default(sample_config):
    """Test that missing keys return default value."""
    value = sample_config.get('nonexistent.key', 'default')
    assert value == 'default'

def test_config_get_missing_key_returns_none(sample_config):
    """Test that missing keys return None if no default."""
    value = sample_config.get('nonexistent.key')
    assert value is None

def test_config_set_value(sample_config):
    """Test setting configuration values at runtime."""
    sample_config.set('test.new_key', 'new_value')
    assert sample_config.get('test.new_key') == 'new_value'
```

**Running Tests:**

```bash
# Install dev dependencies
uv sync --extra dev

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=portraits --cov-report=html

# Run specific test file
uv run pytest tests/test_utils.py

# Run with verbose output
uv run pytest -v
```

---

### 3.3 Write Proper README.md

**File:** `README.md`

```markdown
# Portraits: Multimodal AI Generation Suite

Generate professional images, videos, speech, and morphing effects using state-of-the-art AI models.

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- 🖼️ **Image Generation** - SDXL Turbo with custom LoRA weights support
- 🎬 **Text-to-Video** - SkyReels-V2 (14B parameters, 540P quality)
- 🎙️ **Text-to-Speech** - Maya1 (3B parameters, expressive voices with emotion tags)
- 🔄 **Video Morphing** - MediaPipe facial landmark-based morphing
- 🌐 **Web UI** - Gradio interface for all features

## Quick Start

### Installation

Install with [uv](https://github.com/astral-sh/uv) (recommended):

```bash
# Clone repository
git clone https://github.com/yourusername/portraits.git
cd portraits

# Install for image generation only
uv sync --extra image

# Or install all features
uv sync --extra all
```

### Usage

**Command Line:**

```bash
# Generate an image
portraits image --prompt "professional headshot of a woman"

# Generate a video
portraits video --prompt "cinematic drone shot over mountains"

# Generate speech
portraits voice --text "Hello world!" --voice "warm, friendly"

# Create morphing video
portraits morph --input images/ --output morphed.mp4

# Launch web UI
portraits ui
```

**Python API:**

```python
from generators import generate_headshot, generate_video, generate_voice

# Generate image
image_path = generate_headshot(
    prompt="professional headshot",
    num_images=1
)

# Generate video
video_path = generate_video(
    prompt="sunset over ocean",
    num_frames=97
)

# Generate speech
audio_path = generate_voice(
    text="Welcome to Portraits!",
    description="warm, friendly, medium pitch"
)
```

## Installation Options

Install only the features you need:

```bash
# Image generation (SDXL Turbo + LoRA)
uv sync --extra image

# Text-to-video (SkyReels-V2)
uv sync --extra video

# Text-to-speech (Maya1)
uv sync --extra voice

# Video morphing (MediaPipe)
uv sync --extra morph

# Web interface
uv sync --extra ui

# Everything
uv sync --extra all

# Development (includes testing tools)
uv sync --extra dev
```

## Requirements

- Python 3.11+
- 8GB+ VRAM recommended (GPU acceleration)
- Supports: CUDA, Apple Silicon (MPS), CPU

## Documentation

See [CLAUDE.md](CLAUDE.md) for comprehensive documentation including:
- Detailed usage examples
- Architecture notes
- Customization guide
- Troubleshooting
- API reference

## Examples

### Image Generation

```bash
# Basic generation
portraits image --prompt "portrait of a woman"

# With LoRA weights
portraits image --prompt "ink drawing style portrait" \
  --lora-path ./Ink_drawing_style.safetensors \
  --lora-scale 0.8

# Batch generation
portraits image --prompt "professional headshot" --num-images 10
```

### Text-to-Video

```bash
# Generate 4-second video
portraits video --prompt "cinematic shot of a waterfall"

# Longer video with higher quality
portraits video --prompt "underwater coral reef" \
  --num-frames 145 \
  --guidance-scale 7.0
```

### Text-to-Speech

```bash
# Basic speech
portraits voice --text "Hello world!"

# With voice customization
portraits voice --text "Welcome to the show!" \
  --voice "energetic, young, high pitch"

# With emotion tags
portraits voice --text "That's amazing! <laugh> <excited>"
```

### Video Morphing

```bash
# Create morphing video from images
portraits morph --input output/ --output morphed.mp4

# Customize morphing speed
portraits morph --input images/ --output result.mp4 \
  --fps 30 --morph-frames 12
```

## Hardware Recommendations

- **Minimum:** CPU only (slow but functional)
- **Recommended:** GPU with 16GB+ VRAM
- **Optimal:** NVIDIA GPU with 24GB+ VRAM (A100, RTX 4090)
- **Apple Silicon:** M1/M2/M3 with 16GB+ unified memory

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

- [SDXL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) by Stability AI
- [SkyReels-V2](https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-540P) by Skywork AI
- [Maya1](https://huggingface.co/maya-research/maya1) by Maya Research
- [MediaPipe](https://github.com/google/mediapipe) by Google
```

---

### 3.4 Configure Code Quality Tools

**Add to pyproject.toml (already included in Phase 2.3):**

```toml
[tool.black]
line-length = 100
target-version = ['py311']
exclude = '''
/(
    \.git
  | \.venv
  | build
  | dist
)/
'''

[tool.ruff]
line-length = 100
target-version = "py311"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort (import sorting)
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
]
ignore = [
    "E501",  # line too long (handled by black)
]

[tool.ruff.isort]
known-first-party = ["core", "generators", "ui"]
section-order = ["future", "standard-library", "third-party", "first-party", "local-folder"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
disallow_untyped_defs = false  # Enable gradually
```

**Usage:**

```bash
# Install dev tools
uv sync --extra dev

# Format code
uv run black portraits/

# Check code quality
uv run ruff check portraits/

# Auto-fix issues
uv run ruff check --fix portraits/

# Type checking
uv run mypy portraits/
```

---

### 3.5 Refactor Long Functions

**Target:** `generate_headshot()` in `portraits/generators/image.py`

**Before:** 98-line monolithic function

**After:** Split into focused functions

```python
# portraits/generators/image.py

def _load_sdxl_pipeline(dtype: torch.dtype) -> AutoPipelineForText2Image:
    """Load SDXL Turbo pipeline.

    Args:
        dtype: Torch dtype for model weights

    Returns:
        AutoPipelineForText2Image: Loaded pipeline
    """
    model_id = config.get('models.sdxl_turbo', 'stabilityai/sdxl-turbo')

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        dtype=dtype,
        variant="fp16"
    )

    # Keep VAE in float32 for quality
    if hasattr(pipe, 'upcast_vae'):
        pipe.upcast_vae()

    return pipe

def _load_lora_weights(
    pipe: AutoPipelineForText2Image,
    lora_path: str,
    lora_scale: float
) -> None:
    """Load LoRA weights into pipeline.

    Args:
        pipe: Pipeline to load LoRA into
        lora_path: Path to LoRA weights
        lora_scale: Adapter strength

    Raises:
        GenerationError: If LoRA loading fails critically
    """
    try:
        print(f"Loading LoRA from: {lora_path}")
        pipe.load_lora_weights(lora_path, adapter_name="drawing")

        if hasattr(pipe, 'set_adapters'):
            pipe.set_adapters(["drawing"], adapter_weights=[lora_scale])

        print(f"✓ LoRA loaded with scale: {lora_scale}")

    except Exception as e:
        print(f"⚠ Warning: LoRA loading failed: {e}")
        print("  Continuing without LoRA...")

def _generate_single_image(
    pipe: AutoPipelineForText2Image,
    prompt: str,
    params: dict
) -> Image.Image:
    """Generate a single image.

    Args:
        pipe: Configured pipeline
        prompt: Text prompt
        params: Generation parameters (width, height, etc.)

    Returns:
        PIL.Image: Generated image
    """
    image = pipe(
        prompt=prompt,
        num_inference_steps=params.get('num_inference_steps', 2),
        guidance_scale=params.get('guidance_scale', -100),
        width=params.get('width', 512),
        height=params.get('height', 512),
        clip_skip=params.get('clip_skip', 1)
    ).images[0]

    return image

def generate_headshot(
    prompt: str = "",
    output_dir: str = "output",
    width: int = 512,
    height: int = 512,
    num_images: int = 1,
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
    clip_skip: int = 1,
    guidance_scale: float = -100
) -> Union[str, List[str]]:
    """Generate professional headshots using SDXL Turbo.

    This is now a clean orchestrator function that delegates to helpers.

    Args:
        prompt: Text prompt for generation
        output_dir: Output directory path
        width: Image width
        height: Image height
        num_images: Number of images to generate
        lora_path: Optional LoRA weights path
        lora_scale: LoRA strength (0.0-1.0)
        clip_skip: CLIP layers to skip
        guidance_scale: CFG scale (-100 disables for Turbo)

    Returns:
        str or list: Path(s) to generated image(s)
    """
    # Setup
    device, dtype = get_device_and_dtype()
    output_path = ensure_output_dir(output_dir)

    # Load pipeline
    print("Loading SDXL Turbo model...")
    pipe = _load_sdxl_pipeline(dtype)
    pipe = pipe.to(device)

    # Load LoRA if specified
    if lora_path:
        _load_lora_weights(pipe, lora_path, lora_scale)

    # Generation parameters
    params = {
        'width': width,
        'height': height,
        'num_inference_steps': 2,
        'guidance_scale': guidance_scale,
        'clip_skip': clip_skip,
    }

    # Generate images
    print(f"Generating {num_images} image(s) with prompt: '{prompt}'")
    generated_paths = []

    for i in range(num_images):
        print(f"Generating image {i+1}/{num_images}...")

        image = _generate_single_image(pipe, prompt, params)

        # Save image
        filename = get_output_filename("headshot", "png", index=i if num_images > 1 else None)
        filepath = output_path / filename
        image.save(filepath)

        generated_paths.append(str(filepath))
        print(f"  ✓ Saved: {filepath}")

    print(f"\n✓ Successfully generated {len(generated_paths)} image(s)")
    return generated_paths if num_images > 1 else generated_paths[0]
```

**Benefits:**
- Each function has single responsibility
- Easier to test individual components
- More maintainable
- Better code reuse

---

## Phase 4: Testing & Validation (1 hour)

### 4.1 Test Checklist

**Structural Tests:**
- [ ] All imports work correctly
- [ ] Module structure is valid
- [ ] CLI entry point works
- [ ] Config file loads successfully

**Functional Tests:**
- [ ] Device detection works on current hardware
- [ ] Utility functions work as expected
- [ ] Config values are accessible
- [ ] All unit tests pass

**Integration Tests:**
- [ ] Image generation works end-to-end
- [ ] Video generation works (if model downloaded)
- [ ] Voice generation works
- [ ] Morphing works with sample images
- [ ] Web UI launches successfully

### 4.2 Test Commands

```bash
# 1. Run unit tests
uv run pytest tests/ -v

# 2. Test CLI
uv run python -m portraits --help
uv run python -m portraits image --help
uv run python -m portraits video --help

# 3. Test imports
uv run python -c "from core import config, get_device_and_dtype; print('OK')"
uv run python -c "from generators import generate_headshot; print('OK')"

# 4. Test configuration
uv run python -c "from core.config import config; print(config.get('models.sdxl_turbo'))"

# 5. Test device detection
uv run python -c "from core.device import get_device_and_dtype; print(get_device_and_dtype())"

# 6. Code quality checks
uv run ruff check portraits/
uv run black --check portraits/
uv run mypy portraits/

# 7. Test image generation (if dependencies installed)
uv run python -m portraits image --prompt "test image" --num-images 1
```

### 4.3 Validation Criteria

**All checks must pass:**
- ✅ Unit tests: 100% pass rate
- ✅ Import tests: No errors
- ✅ CLI help: Displays correctly
- ✅ Code quality: Ruff/Black/MyPy clean
- ✅ At least one end-to-end generation works

---

## Summary

### Files Created/Modified

**New Files (11):**
1. `portraits/core/device.py` - Device detection
2. `portraits/core/utils.py` - Shared utilities
3. `portraits/core/config.py` - Configuration management
4. `portraits/core/exceptions.py` - Custom exceptions
5. `portraits/core/__init__.py` - Core module exports
6. `portraits/generators/__init__.py` - Generator exports
7. `portraits/ui/__init__.py` - UI exports
8. `portraits/tests/test_device.py` - Device tests
9. `portraits/tests/test_utils.py` - Utility tests
10. `portraits/tests/test_config.py` - Config tests
11. `config.yaml` - Configuration file

**Modified Files (7):**
1. `main.py` - Full CLI implementation
2. `pyproject.toml` - Optional dependencies, scripts, tooling
3. `portraits/generators/image.py` - Use shared utilities
4. `portraits/generators/video.py` - Remove duplicate code
5. `portraits/generators/voice.py` - Remove duplicate code
6. `portraits/ui/app.py` - Update imports
7. `README.md` - Comprehensive documentation

**Moved Files (5):**
1. `generate_images.py` → `portraits/generators/image.py`
2. `generate_video.py` → `portraits/generators/video.py`
3. `generate_voice.py` → `portraits/generators/voice.py`
4. `generate_morph_video.py` → `portraits/generators/morph.py`
5. `app.py` → `portraits/ui/app.py`

### Expected Outcomes

**Code Metrics:**
- **Lines eliminated:** ~150 (from duplication)
- **Files added:** 11 (utilities + tests)
- **Test coverage:** >80%
- **Code quality:** Ruff/Black/MyPy clean

**User Benefits:**
- Faster installation (5 vs 15 packages for specific features)
- Professional CLI interface
- Better organized codebase
- Easier to maintain and extend
- Type hints for better IDE support

**Developer Benefits:**
- Centralized configuration
- Reusable utilities
- Better error messages
- Comprehensive tests
- Standard Python package structure

### Time Estimate

- **Phase 1 (Critical):** 1-2 hours
- **Phase 2 (Structural):** 2-3 hours
- **Phase 3 (Quality):** 1-2 hours
- **Phase 4 (Testing):** 1 hour

**Total:** 5-8 hours for complete implementation

---

## Next Steps

1. **Review this plan** and approve changes
2. **Backup current code** (git commit)
3. **Execute Phase 1** (critical fixes)
4. **Test Phase 1** before proceeding
5. **Execute Phases 2-3** (structure + quality)
6. **Run full test suite**
7. **Create pull request** with changes
8. **Update documentation** as needed

---

**Questions or concerns?** Review each phase and identify any blockers before starting implementation.
