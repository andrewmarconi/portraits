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
portraits voice --text "Hello world!" --voice "warm, friendly, medium pitch"

# Create morphing video
portraits morph --input images/ --output morphed.mp4

# Launch web UI
portraits ui
```

**Python API:**

```python
from portraits import generate_headshot, generate_video, generate_voice, create_mesh_morphing_video

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

# Create morphing video
morph_path = create_mesh_morphing_video(
    input_dir="images/",
    output_file="morphed.mp4"
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
portraits voice --text "Welcome to the show! <excited>" \
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

MIT License - see [LICENSE](LICENSE) for details

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

- [SDXL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) by Stability AI
- [SkyReels-V2](https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-540P) by Skywork AI
- [Maya1](https://huggingface.co/maya-research/maya1) by Maya Research
- [MediaPipe](https://github.com/google/mediapipe) by Google