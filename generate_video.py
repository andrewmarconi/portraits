#!/usr/bin/env python3
"""
Text-to-video generation using Skywork/SkyReels-V2-T2V-14B-540P

This script generates videos from text prompts using the SkyReels-V2 model.

Installation:
    1. Install SkyReels-V2 pipeline code:
       git clone https://github.com/SkyworkAI/SkyReels-V2
       cd SkyReels-V2
       pip install -r requirements.txt
       cd ..

    2. Download model weights (~28GB):
       git lfs install
       git clone https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-540P

    3. Install Python dependencies:
       uv sync

Usage:
    uv run python generate_video.py

Requirements:
    - VRAM: ~43.4GB for 540P generation (use --offload for lower VRAM)
    - Python 3.11+
    - Torch with CUDA/MPS support recommended

Device Support:
    - CUDA: Full support, recommended for best performance (24GB+ VRAM)
    - MPS (Apple Silicon): Supported with CPU offloading enabled
    - CPU: Supported but very slow (30+ minutes per video)

MPS-Specific Notes:
    - Enable CPU offloading (default) for better stability
    - May require PYTORCH_ENABLE_MPS_FALLBACK=1 environment variable
    - Works best with PyTorch 2.0+ with MPS backend
    - Unified memory architecture benefits from dynamic allocation
"""

import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
import warnings

# Enable MPS fallback for unsupported operations (Apple Silicon)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Suppress warnings
warnings.filterwarnings("ignore")


def get_device_and_dtype():
    """Auto-detect optimal device and dtype for the current system."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print(f"✓ Using CUDA GPU with bfloat16")
        # Check VRAM
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  Available VRAM: {vram_gb:.1f}GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print(f"✓ Using Apple Silicon (MPS) with float16")
        # Check PyTorch version for MPS compatibility
        torch_version = torch.__version__.split('+')[0]  # Remove +cpu/+cu118 suffix
        print(f"  PyTorch version: {torch_version}")
        major, minor = map(int, torch_version.split('.')[:2])
        if major < 2:
            print(f"  ⚠ Warning: PyTorch {torch_version} has limited MPS support")
            print(f"  Recommendation: Upgrade to PyTorch 2.0+ for better MPS performance")
        print(f"  Note: MPS support for large video models may be experimental")
        print(f"  Recommendation: Enable CPU offloading (default) for stability")
    else:
        device = "cpu"
        dtype = torch.float16
        print(f"⚠ Using CPU with float16 (will be slow)")
        print(f"  Note: Video generation on CPU can take 30+ minutes")

    return device, dtype


def load_pipeline(model_id: str, device: str, dtype: torch.dtype, enable_offload: bool = True):
    """
    Load the SkyReels-V2 text-to-video pipeline.

    Args:
        model_id: Hugging Face model ID or local path
        device: Target device (cuda/mps/cpu)
        dtype: Model dtype (bfloat16/float16)
        enable_offload: Enable CPU offloading to reduce VRAM usage

    Returns:
        Loaded pipeline ready for inference
    """
    try:
        from diffusers import SkyReelsV2DiffusionForcingPipeline
    except ImportError:
        raise ImportError(
            "SkyReels-V2 pipeline not found. Please install from:\n"
            "  git clone https://github.com/SkyworkAI/SkyReels-V2\n"
            "  cd SkyReels-V2\n"
            "  pip install -r requirements.txt"
        )

    print(f"Loading model from: {model_id}")
    print("⚠ This is a 14B parameter model - initial download may be large (~28GB)")

    # Try to load the pipeline with better error handling
    try:
        pipe = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "Entry Not Found" in error_msg:
            raise RuntimeError(
                f"Model not found at '{model_id}'.\n\n"
                "The SkyReels-V2 model requires manual download. Please:\n"
                "  1. Clone the model repository:\n"
                "     git lfs install\n"
                "     git clone https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-540P\n\n"
                "  2. Update model_id in the script to the local path:\n"
                "     model_id = './SkyReels-V2-T2V-14B-540P'\n\n"
                "  Alternative: Use the official SkyReels-V2 repository's generate_video.py script\n"
                "  from: https://github.com/SkyworkAI/SkyReels-V2"
            ) from e
        raise

    # Enable memory optimizations based on device
    if enable_offload and device != "cpu":
        if device == "mps":
            print("✓ Enabling CPU offloading for MPS (reduces memory pressure)")
            # MPS benefits from CPU offloading due to unified memory architecture
            try:
                pipe.enable_model_cpu_offload()
            except Exception as e:
                print(f"⚠ CPU offloading failed on MPS, using direct device transfer: {e}")
                pipe = pipe.to(device)
        else:
            # CUDA device
            print("✓ Enabling CPU offloading to reduce VRAM usage")
            pipe.enable_model_cpu_offload()
    else:
        print(f"✓ Loading model directly to {device}")
        pipe = pipe.to(device)

    # Enable additional memory optimizations
    # These work across CUDA, MPS, and CPU
    if hasattr(pipe, 'enable_attention_slicing'):
        try:
            pipe.enable_attention_slicing()
            print("✓ Enabled attention slicing")
        except Exception as e:
            print(f"⚠ Could not enable attention slicing: {e}")

    if hasattr(pipe, 'enable_vae_slicing'):
        try:
            pipe.enable_vae_slicing()
            print("✓ Enabled VAE slicing")
        except Exception as e:
            print(f"⚠ Could not enable VAE slicing: {e}")

    # MPS-specific optimizations
    if device == "mps":
        # Set PyTorch to use less memory for MPS
        try:
            torch.mps.set_per_process_memory_fraction(0.0)  # Allow dynamic allocation
            print("✓ Configured MPS for dynamic memory allocation")
        except Exception:
            pass  # Older PyTorch versions may not have this

    return pipe


def generate_video(
    prompt: str,
    model_id: str = "./SkyReels-V2-T2V-14B-540P",
    num_frames: int = 97,
    width: int = 960,
    height: int = 544,
    fps: int = 24,
    guidance_scale: float = 6.0,
    num_inference_steps: int = 50,
    enable_offload: bool = True,
    output_dir: str = "output",
) -> str:
    """
    Generate a video from a text prompt using SkyReels-V2.

    Args:
        prompt: Text description of the video to generate
        model_id: Local path to model or HuggingFace ID (default: ./SkyReels-V2-T2V-14B-540P)
                  Must be downloaded first with: git clone https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-540P
        num_frames: Number of frames to generate (default: 97 for ~4 seconds at 24fps)
        width: Video width in pixels (default: 960 for 540P)
        height: Video height in pixels (default: 544 for 540P)
        fps: Frames per second for output video (default: 24)
        guidance_scale: CFG scale for prompt adherence (default: 6.0)
        num_inference_steps: Number of denoising steps (default: 50)
        enable_offload: Enable CPU offloading to reduce VRAM (default: True)
        output_dir: Directory to save generated videos (default: "output")

    Returns:
        Path to the generated video file
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Get device and dtype
    device, dtype = get_device_and_dtype()

    # Load pipeline
    pipe = load_pipeline(model_id, device, dtype, enable_offload)

    print(f"\n{'='*60}")
    print(f"Generating video...")
    print(f"Prompt: {prompt}")
    print(f"Resolution: {width}x{height}")
    print(f"Frames: {num_frames} ({num_frames/fps:.1f}s at {fps}fps)")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"{'='*60}\n")

    # Generate video
    try:
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        # Extract frames from output
        frames = output.frames[0]

    except RuntimeError as e:
        error_msg = str(e)
        # Check for MPS-specific errors
        if device == "mps" and ("MPS" in error_msg or "mps" in error_msg):
            print(f"❌ MPS-specific error encountered: {e}")
            print("\n⚠ Troubleshooting suggestions:")
            print("  1. Try reducing num_frames (e.g., 49 instead of 97)")
            print("  2. Ensure you have the latest PyTorch with MPS support")
            print("  3. Try setting PYTORCH_ENABLE_MPS_FALLBACK=1")
            print("  4. Consider using CUDA if available, or CPU as fallback")
        else:
            print(f"❌ Error during generation: {e}")
        raise
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise

    # Save video using PIL and export_to_video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"video_{timestamp}.mp4"
    video_path = output_path / video_filename

    # Try to use diffusers export utility
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, str(video_path), fps=fps)
        print(f"✓ Video saved using diffusers export: {video_path}")
    except ImportError:
        # Fallback to manual export using opencv
        import cv2

        print("⚠ Using OpenCV fallback for video export")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            fps,
            (width, height)
        )

        for frame in frames:
            # Convert PIL Image to numpy array (RGB to BGR for OpenCV)
            if isinstance(frame, Image.Image):
                frame_np = np.array(frame)
            else:
                frame_np = frame

            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        print(f"✓ Video saved using OpenCV: {video_path}")

    # Print video info
    duration = num_frames / fps
    print(f"\n{'='*60}")
    print(f"✓ Generation complete!")
    print(f"Duration: {duration:.1f} seconds")
    print(f"File: {video_path}")
    print(f"{'='*60}\n")

    return str(video_path)


def main():
    """Main entry point with example usage."""

    # IMPORTANT: Download the model first!
    # Run this command before using the script:
    #   git lfs install
    #   git clone https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-540P
    #
    # This will download ~28GB of model files to ./SkyReels-V2-T2V-14B-540P/

    # Example prompts - modify these or add your own
    prompts = [
        # "A cinematic shot of a majestic mountain landscape at sunrise, with golden light rays breaking through clouds, camera slowly panning across snow-capped peaks",
        "A close-up of a delicate drag queen, zooming out to a warmly lit cabaret-style room full of people",
        # "An aerial view of ocean waves crashing on a sandy beach, foam spreading across the shore",
    ]

    # Generation settings
    # Note: Device is auto-detected (CUDA > MPS > CPU)
    # MPS users: Keep enable_offload=True for best stability
    settings = {
        "model_id": "./SkyReels-V2-T2V-14B-540P",  # Local path to downloaded model
        "num_frames": 97,           # ~4 seconds at 24fps (97/24 ≈ 4.04s)
                                     # For MPS with limited memory, try 49 frames (~2s)
        "width": 960,               # 540P resolution
        "height": 544,
        "fps": 24,
        "guidance_scale": 6.0,      # CFG scale - higher = more prompt adherence
        "num_inference_steps": 50,  # More steps = higher quality but slower
        "enable_offload": True,     # Essential for <43GB VRAM systems (CUDA/MPS)
    }

    print("SkyReels-V2 Text-to-Video Generator")
    print("=" * 60)
    print(f"Generating {len(prompts)} video(s)...")
    print()

    # Generate videos for each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Processing prompt...")
        try:
            video_path = generate_video(prompt=prompt, **settings)
            print(f"✓ Video {i} saved to: {video_path}")
        except Exception as e:
            print(f"❌ Failed to generate video {i}: {e}")
            continue

    print("\n✓ All videos generated successfully!")


if __name__ == "__main__":
    main()
