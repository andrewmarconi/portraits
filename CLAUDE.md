# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multimodal AI generation project that supports:
1. **Image generation** using SDXL Turbo with custom LoRA weights
2. **Text-to-video generation** using SkyReels-V2 (14B parameter model)
3. **Video morphing** from generated images using OpenCV and MediaPipe
4. **Text-to-speech** using Maya1 (3B parameter voice model)

**Key Technologies:**
- SDXL Turbo via Hugging Face Diffusers
- SkyReels-V2-T2V-14B-540P (text-to-video generation)
- Maya1 (text-to-speech with SNAC codec)
- PyTorch with multi-device support (CUDA, MPS/Apple Silicon, CPU)
- OpenCV 4.x for video processing
- MediaPipe for facial landmark detection
- SNAC neural audio codec for voice synthesis
- NumPy for numerical operations
- `uv` for package management (modern Python package manager)

## Package Management

This project uses **uv** instead of pip/conda. All dependency management is in `pyproject.toml`.

**Install dependencies:**
```bash
uv sync
```

**Run scripts:**
```bash
uv run python generate_headshot.py
uv run python create_mesh_morphing.py
```

## Main Components

### 1. Image Generation (`generate_headshot.py`)

The primary script for generating images using SDXL Turbo with optional LoRA weights.

**Key Function:** `generate_headshot()`
- **prompt** (str): Text prompt for generation
- **num_images** (int): Number of images to generate
- **lora_path** (str): Path to LoRA weights (.safetensors file or HuggingFace repo)
- **lora_scale** (float): LoRA adapter strength (0.0-1.0)
- **width/height** (int): Image dimensions
- **clip_skip** (int): CLIP layers to skip
- **guidance_scale** (float): CFG scale (-100 disables guidance for Turbo)

**Architecture Notes:**
- Auto-detects device: CUDA → bfloat16, MPS → float16, CPU → float16
- VAE stays in float32 for quality (`upcast_vae()`)
- Uses 2 inference steps (SDXL Turbo optimized)
- Saves images to `output/` with timestamps

### 2. Mesh Morphing (`create_mesh_morphing.py`) **[RECOMMENDED]**

The primary script for creating professional facial morphing videos using MediaPipe Face Mesh.

**Key Technology:**
- **MediaPipe Face Mesh**: Detects 468 facial landmarks per face
- **Delaunay Triangulation**: Creates mesh from landmarks using scipy
- **Affine Warping**: Each triangle is independently warped from source to destination
- **Smooth Interpolation**: Vertices smoothly transition between frames

**Key Function:** `create_mesh_morphing_video()`
- **input_dir** (str): Directory containing source images
- **output_file** (str): Output video path
- **fps** (int): Frames per second (default: 24)
- **morph_frames** (int): Number of interpolation frames between images (default: 8)
  - Total frames = N + (N-1) × morph_frames
  - Example: 10 images with morph_frames=8 → 10 + (9 × 8) = 82 frames
  - Modify at line 316 in `__main__` block
- **visualize_landmarks** (bool): Draw landmarks on frames for debugging

**Architecture:**
1. Detects 468 facial landmarks in each image using MediaPipe
2. Adds boundary points (corners + edge midpoints) for full coverage
3. Creates Delaunay triangulation from landmarks + boundary points
4. Interpolates landmark positions for each morph frame
5. Warps each triangle using affine transformation
6. Blends warped triangles into final morphed frame

**When to use:**
- For images with faces (portraits, headshots, character art)
- When you want professional smooth morphing effects
- When facial features should flow naturally between frames

**Fallback Behavior:**
- If face detection fails on any image pair, falls back to simple crossfade
- MediaPipe requires confidence threshold: min_detection_confidence=0.5
- Works best with clear, front-facing faces

### 3. Text-to-Video Generation (`generate_video.py`)

Generate videos directly from text prompts using the SkyReels-V2-T2V-14B-540P model.

**Key Function:** `generate_video()`
- **prompt** (str): Text description of the video to generate
- **num_frames** (int): Number of frames (default: 97 for ~4 seconds at 24fps)
- **width/height** (int): Video dimensions (default: 960×544 for 540P)
- **fps** (int): Frames per second (default: 24)
- **guidance_scale** (float): CFG scale for prompt adherence (default: 6.0)
- **num_inference_steps** (int): Denoising steps (default: 50)
- **enable_offload** (bool): CPU offloading to reduce VRAM (default: True)

**Architecture Notes:**
- 14B parameter diffusion model for high-quality video generation
- Generates 540P (960×544) videos natively
- Requires ~43.4GB VRAM without offloading
- Uses CPU offloading by default to work with lower VRAM GPUs
- Auto-detects device: CUDA → bfloat16, MPS → float16, CPU → float16
- Supports attention slicing and VAE slicing for memory optimization
- **MPS (Apple Silicon) optimizations:**
  - Automatic MPS fallback enabled for unsupported operations
  - Dynamic memory allocation for unified memory architecture
  - CPU offloading with error handling specific to MPS
  - PyTorch version checking for compatibility warnings

**Installation Requirements:**
1. Install the SkyReels-V2 pipeline code:
```bash
git clone https://github.com/SkyworkAI/SkyReels-V2
cd SkyReels-V2
pip install -r requirements.txt
cd ..
```

2. Download the model weights (~28GB):
```bash
git lfs install
git clone https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-540P
```

3. Install additional dependencies via uv:
```bash
uv add sentencepiece protobuf ftfy
# Or sync all dependencies
uv sync
```

**Important:** The model must be downloaded locally before running generate_video.py. The script expects the model at `./SkyReels-V2-T2V-14B-540P/`

**When to use:**
- Generate videos from text descriptions (cinematic shots, nature scenes, animations)
- Create video content without source images
- Prototype video ideas quickly
- Generate ~4 second video clips (97 frames at 24fps)

### 4. Text-to-Speech Generation (`generate_voice.py`)

Generate natural-sounding speech from text using the Maya1 model with SNAC neural codec.

**Key Function:** `generate_voice()`
- **text** (str): Text to convert to speech
- **description** (str): Natural language voice description (e.g., "30-year-old, friendly, medium pitch")
- **temperature** (float): Sampling temperature for variation (default: 0.4)
- **top_p** (float): Nucleus sampling threshold (default: 0.9)
- **max_tokens** (int): Maximum audio tokens to generate (default: 2048)

**Architecture Notes:**
- 3B parameter decoder-only transformer (Llama-style backbone)
- Uses SNAC neural codec for hierarchical audio encoding
- Generates 24 kHz mono audio
- Supports 20+ emotion tags: `<laugh>`, `<cry>`, `<whisper>`, `<angry>`, `<gasp>`, etc.
- Sub-100ms latency streaming capability (with vLLM deployment)
- Multi-accent English support
- Auto-detects device: CUDA → bfloat16, MPS → float16, CPU → float16

**Voice Description Examples:**
- `"40-year-old, warm, low pitch, conversational"`
- `"young female, energetic, high pitch"`
- `"elderly male, calm, deep voice, thoughtful"`
- `"middle-aged, professional, medium pitch, clear"`

**Emotion Tags:**
Embed emotion tags directly in text for expressive speech:
```python
"Hello there! <laugh> That's amazing! <excited>"
"I'm so sorry to hear that. <sad> <whisper> It'll be okay."
```

**Installation Requirements:**
1. Install dependencies:
```bash
uv add snac soundfile
uv sync
```

2. Models are downloaded automatically from HuggingFace:
   - `maya-research/maya1` (~6GB)
   - `hubertsiuzdak/snac_24khz` (SNAC decoder)

**Hardware Requirements:**
- Recommended: 16GB+ VRAM (A100, H100, RTX 4090)
- Works on: CUDA, MPS (Apple Silicon), CPU
- Generation time: ~10-30 seconds per sentence (GPU)

**When to use:**
- Generate voiceovers for videos
- Create narration for content
- Prototype voice interfaces
- Generate expressive speech with emotions
- Multi-accent voice synthesis

## Critical Technical Details

### MediaPipe Face Mesh Configuration

The face mesh detector is initialized with specific settings:
```python
mp_face_mesh.FaceMesh(
    static_image_mode=True,        # Treat each image independently
    max_num_faces=1,                # Detect only the primary face
    refine_landmarks=True,          # Include iris landmarks
    min_detection_confidence=0.5,   # Detection threshold
    min_tracking_confidence=0.5     # Tracking threshold
)
```

**Important Notes:**
- Detects 468 landmarks when `refine_landmarks=True` (includes iris)
- Only the first detected face is used if multiple faces present
- Images are converted BGR→RGB for MediaPipe processing
- Landmark coordinates are normalized (0-1) then converted to pixels

### NumPy API Compatibility

Modern NumPy requires correct usage of `np.unravel_index()`:

**Correct pattern:**
```python
# First find the flat index
max_idx = np.argmax(array)
# Then convert to multi-dimensional coordinates
y, x = np.unravel_index(max_idx, array.shape)
```

**Incorrect (will error):**
```python
# WRONG - unravel_index needs an index, not an array
y, x = np.unravel_index(array)
```

This pattern appears in motion estimation code (FFT cross-correlation, SAD matching).

### OpenCV 4.x API

Use modern OpenCV API:
```python
# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Modern API
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# NOT: cv2.VideoWriter.fourcc() - old API
```

### Device and Dtype Selection

The image generation pipeline auto-selects optimal dtype per device:
- **CUDA**: `torch.bfloat16` (better for modern GPUs)
- **MPS** (Apple Silicon): `torch.float16` (required for MPS)
- **CPU**: `torch.float16` (fallback)

VAE always uses float32 for quality preservation.

## Workflow

**Typical usage patterns:**

### Workflow A: Image Generation → Morphing Video

1. **Generate images:**
   ```bash
   uv run python generate_headshot.py
   ```
   Modify the `__main__` block to adjust prompt, LoRA path, and count.

2. **Create morphing video from generated images:**
   ```bash
   uv run python create_mesh_morphing.py
   ```
   This creates a professional mesh-based morphing video using facial landmarks.

3. **Check output:**
   - Generated images: `output/headshot_*.png`
   - Generated videos: `output/*.mp4`

### Workflow B: Direct Text-to-Video Generation

1. **Generate video from text prompt:**
   ```bash
   uv run python generate_video.py
   ```
   Modify the `prompts` list in the `__main__` block to customize video content.

2. **Check output:**
   - Generated videos: `output/video_*.mp4`

### Workflow C: Text-to-Speech Generation

1. **Generate voice from text:**
   ```bash
   uv run python generate_voice.py
   ```
   Modify the `examples` list in the `__main__` block to customize voice samples.

2. **Check output:**
   - Generated audio: `output/voice_*.wav`

## Customization Examples

**Adjust morphing speed:**
```python
# In create_mesh_morphing.py, line 316
create_mesh_morphing_video(
    morph_frames=12,  # More frames = slower, smoother transition
    fps=24            # Higher fps = faster playback
)
```

**Debug face detection:**
```python
# Enable landmark visualization
create_mesh_morphing_video(
    visualize_landmarks=True  # Green dots show detected landmarks
)
```

**Change image generation settings:**
```python
# In generate_headshot.py, line 125
result_path = generate_headshot(
    num_images=32,    # Generate more/fewer images
    prompt="your custom prompt",
    lora_scale=0.7,   # Adjust LoRA strength (0.0-1.0)
    clip_skip=2       # Try different CLIP skip values
)
```

**Adjust text-to-video generation:**
```python
# In generate_video.py, modify settings dict
settings = {
    "num_frames": 145,           # Longer video (145/24 ≈ 6 seconds)
    "guidance_scale": 7.0,       # Higher = more prompt adherence
    "num_inference_steps": 75,   # More steps = higher quality (slower)
    "enable_offload": True,      # Keep enabled for <43GB VRAM
}
```

**Generate shorter/faster videos:**
```python
# Quick generation with lower quality
settings = {
    "num_frames": 49,            # ~2 seconds at 24fps
    "num_inference_steps": 30,   # Faster but lower quality
    "guidance_scale": 5.0,
}
```

**Customize voice generation:**
```python
# In generate_voice.py
audio_path = generate_voice(
    text="Your text here with <laugh> emotion tags!",
    description="25-year-old, cheerful, high pitch, energetic",
    temperature=0.4,       # Lower = more consistent voice
    top_p=0.9,             # Nucleus sampling
    max_tokens=2048,       # Longer audio
)
```

**Different voice styles:**
```python
# Professional narrator
generate_voice(
    text="Welcome to our presentation.",
    description="middle-aged, professional, medium pitch, clear and confident"
)

# Storyteller voice
generate_voice(
    text="Once upon a time, in a land far away...",
    description="elderly, warm, low pitch, storytelling, gentle"
)

# Excited announcer
generate_voice(
    text="And the winner is... <gasp> <excited> Amazing!",
    description="young adult, dynamic, high energy, enthusiastic"
)
```

## LoRA Weights

LoRA weights customize the image style. The project includes `Ink_drawing_style-000001.safetensors` for ink drawing style.

**Loading LoRA:**
```python
pipe.load_lora_weights(lora_path, adapter_name="drawing")
pipe.set_adapters(["drawing"], adapter_weights=[lora_scale])
```

LoRA can be:
- Local `.safetensors` file (e.g., `./Ink_drawing_style-000001.safetensors`)
- HuggingFace repo (e.g., `"user/lora-name"`)

## Environment Variables

`.env` file sets:
- `HF_HOME`: Hugging Face cache location (for model downloads)

## Troubleshooting

**Face detection fails:**
- Ensure faces are clearly visible and front-facing
- Check image quality (not too blurry or dark)
- Try with `visualize_landmarks=True` to see if landmarks are detected
- Falls back to crossfade automatically if detection fails

**Memory issues during image generation:**
- Reduce `num_images` parameter
- Lower resolution with `width=512, height=512`
- Use CPU if GPU memory is insufficient (slower but works)

**Video codec errors:**
- The script uses 'mp4v' codec (widely compatible)
- If playback issues occur, convert with: `ffmpeg -i input.mp4 -c:v libx264 output_h264.mp4`

**MediaPipe installation issues:**
- MediaPipe requires Python 3.11+
- If import fails, try: `uv sync --reinstall-package mediapipe`

**Text-to-video generation issues:**
- **404 Error / Entry Not Found**: Model not downloaded locally
  ```bash
  # Install git-lfs first (required for large files)
  git lfs install
  # Download the model (~28GB)
  git clone https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-540P
  ```
- **ImportError for SkyReelsV2**: Install the SkyReels-V2 repository:
  ```bash
  git clone https://github.com/SkyworkAI/SkyReels-V2
  cd SkyReels-V2
  pip install -r requirements.txt
  ```
- **Missing module errors (ftfy, sentencepiece, etc.)**: Run `uv sync` to install all dependencies
- **Out of VRAM**: Enable offloading (default) or reduce `num_frames` and resolution
- **Slow generation**: Normal for 14B model - expect several minutes per video
- **MPS (Apple Silicon) specific issues:**
  - Ensure PyTorch 2.0+ is installed for best MPS support
  - Keep CPU offloading enabled (default) for stability
  - If errors occur, try: `export PYTORCH_ENABLE_MPS_FALLBACK=1`
  - Reduce `num_frames` to 49 if memory issues occur
  - MPS uses unified memory - monitor overall system RAM usage
- **CUDA errors**: Model works best on CUDA GPUs with 24GB+ VRAM; enable offloading for smaller GPUs

**Text-to-speech generation issues:**
- **ImportError for snac or soundfile**: Run `uv add snac soundfile` then `uv sync`
- **No valid SNAC tokens generated**:
  - Try adjusting `temperature` (0.3-0.6 range)
  - Ensure text is not too short or too long
  - Check that voice description is reasonable
- **Robotic or unnatural voice**:
  - Lower `temperature` (try 0.3-0.4)
  - Simplify voice description
  - Ensure proper punctuation in text
- **Audio artifacts or glitches**:
  - Increase `max_tokens` if audio cuts off
  - Check VRAM availability (16GB+ recommended)
  - Try regenerating with different random seed
- **Slow generation**:
  - Normal for 3B model on CPU (~1-2 minutes per sentence)
  - GPU significantly faster (~10-30 seconds)
  - MPS (Apple Silicon) moderate speed

## Project Structure

```
Portraits/
├── generate_headshot.py       # Image generation (SDXL Turbo + LoRA)
├── generate_video.py          # Text-to-video generation (SkyReels-V2)
├── generate_voice.py          # Text-to-speech generation (Maya1)
├── create_mesh_morphing.py    # Facial mesh morphing videos
├── app.py                     # Gradio multimodal web interface (coming soon)
├── main.py                    # Placeholder entry point
├── output/                    # Generated images, videos, and audio
├── pyproject.toml             # uv dependencies
├── CLAUDE.md                  # Project documentation for Claude Code
└── *.safetensors             # LoRA weight files
```
