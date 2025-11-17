# Portraits: Multimodal AI Generation Suite

Generate professional images, videos, speech, and morphing effects using state-of-the-art AI models.

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Complexity](https://img.shields.io/badge/complexity-optimized-green)

## 🎯 Overview

Portraits is a comprehensive Python suite for AI-powered content generation, featuring:

- 🖼️ **Image Generation** - Professional headshots and artistic images
- 🎬 **Video Generation** - High-quality text-to-video synthesis  
- 🎙️ **Voice Synthesis** - Natural speech with emotion support
- 🔄 **Video Morphing** - Smooth facial landmark-based transitions
- 🌐 **Web Interface** - User-friendly Gradio UI

## ✨ Key Features

### Image Generation
- **SDXL Turbo** model for fast, high-quality output
- **LoRA support** for custom styles and fine-tuning
- **Professional headshots** with consistent lighting and posing
- **Batch processing** for multiple generations

### Video Generation  
- **SkyReels-V2** (14B parameters) for cinematic quality
- **540P resolution** (960x544) output
- **Configurable duration** up to 6 seconds
- **GPU acceleration** with CPU fallback support

### Voice Synthesis
- **Maya1** (3B parameters) for natural speech
- **Emotion tags** (`<laugh>`, `<cry>`, `<whisper>`, etc.)
- **Voice descriptions** for age, pitch, and style control
- **24kHz mono** audio output

### Video Morphing
- **MediaPipe** facial landmark detection
- **Smooth mesh-based** transitions
- **Customizable speed** and frame interpolation
- **Automatic sorting** and validation

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **8GB+ VRAM** recommended (GPU acceleration)
- **Supports:** CUDA, Apple Silicon (MPS), CPU

### Installation

```bash
# Clone the repository
git clone https://github.com/andrewmarconi/portraits.git
cd portraits

# Install with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### Basic Usage

#### Command Line Interface

```bash
# Launch web interface (recommended)
python main.py

# Command examples
python main.py image --prompt "professional headshot of a woman"
python main.py video --prompt "cinematic drone shot over mountains"
python main.py voice --text "Hello world!" --voice "warm, friendly, medium pitch"
python main.py morph --input images/ --output morphed.mp4
```

#### Python API

```python
from portraits.generators import image, video, voice, morph

# Generate professional headshot
image_path = image.generate_headshot(
    prompt="professional headshot of a woman in business attire",
    num_images=1
)

# Generate video from text
video_path = video.generate_video(
    prompt="cinematic drone shot over mountains at sunset",
    num_frames=97
)

# Generate speech with emotion
audio_path = voice.generate_voice(
    text="Welcome to Portraits! <excited>",
    description="warm, friendly, medium pitch"
)

# Create morphing video
morph_path = morph.create_mesh_morphing_video(
    input_dir="portraits/",
    output_file="morphed_video.mp4"
)
```

## 📋 Detailed Usage

### Image Generation

```python
from portraits.generators.image import generate_headshot

# Basic generation
image_path = generate_headshot(
    prompt="professional headshot of a doctor"
)

# Advanced options
image_path = generate_headshot(
    prompt="professional headshot in office lighting",
    num_images=5,
    seed=42,
    lora_path="./custom_style.safetensors",
    lora_scale=0.8
)
```

### Video Generation

```python
from portraits.generators.video import generate_video

# Standard generation
video_path = generate_video(
    prompt="a serene lake surrounded by mountains"
)

# High-quality settings
video_path = generate_video(
    prompt="underwater coral reef with tropical fish",
    num_frames=145,  # ~6 seconds at 24fps
    guidance_scale=7.0,
    num_inference_steps=50
)
```

### Voice Synthesis

```python
from portraits.generators.voice import generate_voice

# Natural speech
audio_path = generate_voice(
    text="Hello, this is a test of the voice synthesis system."
)

# Character voice with emotions
audio_path = generate_voice(
    text="I'm so excited to meet you! <laugh> This is amazing!",
    description="young female, energetic, high pitch",
    temperature=0.7,
    top_p=0.9
)
```

### Video Morphing

```python
from portraits.generators.morph import create_mesh_morphing_video

# Basic morphing
morph_path = create_mesh_morphing_video(
    input_dir="headshots/",
    output_file="transition.mp4"
)

# Custom settings
morph_path = create_mesh_morphing_video(
    input_dir="portraits/",
    output_file="smooth_morph.mp4",
    fps=30,
    morph_frames=12,
    padding=0.2
)
```

## 🖥️ Web Interface

Launch the comprehensive web UI:

```bash
python main.py
```

Features:
- **Tabbed interface** for each generation type
- **Real-time preview** of results
- **Parameter controls** with sliders and inputs
- **Batch processing** support
- **Download management** for generated content

## ⚙️ Configuration

### Hardware Optimization

The system automatically detects and optimizes for your hardware:

- **CUDA GPUs:** Full acceleration, recommended for best performance
- **Apple Silicon:** MPS support with CPU fallback for stability  
- **CPU:** Functional but significantly slower

### Model Requirements

| Feature | Model | VRAM Required | Disk Space |
|----------|--------|----------------|-------------|
| Image Generation | SDXL Turbo | 8GB+ | 12GB |
| Video Generation | SkyReels-V2 | 24GB+ | 28GB |
| Voice Synthesis | Maya1 | 16GB+ | 6GB |
| Video Morphing | MediaPipe | 4GB+ | 500MB |

### Environment Variables

```bash
# PyTorch optimizations
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Apple Silicon
export CUDA_VISIBLE_DEVICES=0           # Select GPU

# Model paths (optional)
export PORTRAITS_MODELS_DIR="./models"
export PORTRAITS_OUTPUT_DIR="./output"
```

## 🏗️ Architecture

### Project Structure

```
portraits/
├── core/                   # Shared utilities
│   ├── config.py           # Configuration management
│   ├── device.py           # Hardware detection
│   ├── utils.py            # Common utilities
│   └── exceptions.py       # Custom exceptions
├── generators/             # AI generation modules
│   ├── image.py           # Image generation
│   ├── video.py           # Video generation  
│   ├── voice.py           # Voice synthesis
│   ├── morph.py           # Video morphing
│   └── *_helpers.py      # Modular helper functions
├── ui/                    # Web interface
│   └── app.py            # Gradio application
└── tests/                 # Test suite
```

### Code Quality

- **✅ Optimized Complexity:** All functions refactored for maintainability
- **✅ Type Hints:** Full type annotation coverage
- **✅ Error Handling:** Comprehensive exception management
- **✅ Documentation:** Detailed docstrings and examples
- **✅ Testing:** Unit tests for core functionality

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/andrewmarconi/portraits.git
cd portraits

# Install development dependencies
uv sync --extra dev

# Run tests
pytest tests/

# Check code quality
ruff check portraits/
mypy portraits/
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with tests
4. Run quality checks: `ruff check && mypy && pytest`
5. Submit a pull request

## 📚 Examples Gallery

### Professional Headshots
```python
# Corporate headshot
generate_headshot(
    prompt="professional corporate headshot, neutral expression, business attire"
)

# Creative professional
generate_headshot(
    prompt="professional headshot, warm lighting, confident smile"
)
```

### Creative Video
```python
# Cinematic scene
generate_video(
    prompt="cinematic shot of a futuristic city at night with flying cars"
)

# Nature documentary
generate_video(
    prompt="close-up of a hummingbird drinking nectar in slow motion"
)
```

### Voice Characters
```python
# News anchor
generate_voice(
    text="Breaking news: AI technology continues to advance rapidly.",
    description="middle-aged male, authoritative, clear diction"
)

# Friendly assistant  
generate_voice(
    text="How can I help you today? <smile>",
    description="young female, cheerful, medium pitch"
)
```

## 🔧 Troubleshooting

### Common Issues

**Out of Memory Errors:**
- Reduce `num_frames` for video generation
- Enable CPU offloading with `enable_offload=True`
- Use smaller batch sizes

**Model Download Issues:**
- Check internet connection
- Verify Hugging Face authentication for private models
- Ensure sufficient disk space

**Performance Issues:**
- Update GPU drivers
- Use CUDA instead of CPU when available
- Close other GPU-intensive applications

### Getting Help

- **Documentation:** Check inline docstrings and examples
- **Issues:** [GitHub Issues](https://github.com/andrewmarconi/portraits/issues)
- **Discussions:** [GitHub Discussions](https://github.com/andrewmarconi/portraits/discussions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stability AI** - [SDXL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) model
- **Skywork AI** - [SkyReels-V2](https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-540P) model  
- **Maya Research** - [Maya1](https://huggingface.co/maya-research/maya1) model
- **Google** - [MediaPipe](https://github.com/google/mediapipe) framework
- **Gradio** - Web interface framework

## 📈 Performance Benchmarks

| Hardware | Image (1x) | Video (4s) | Voice (10s) | Morph (10 imgs) |
|-----------|---------------|--------------|----------------|------------------|
| RTX 4090 | ~2s | ~45s | ~15s | ~8s |
| RTX 3080 | ~4s | ~90s | ~25s | ~15s |
| M2 Ultra  | ~6s | ~120s | ~30s | ~20s |
| CPU (16GB) | ~45s | ~600s | ~180s | ~90s |

*Benchmarks are approximate and depend on model settings and content complexity.*

---

**Portraits** - Professional AI content generation made accessible. 🎨✨