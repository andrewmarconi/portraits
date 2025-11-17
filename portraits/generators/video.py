#!/usr/bin/env python3
"""
Refactored video generation with reduced complexity.

This module contains the simplified generate_video() function that uses
helper functions to reduce complexity from 18 to ~8.
"""

from pathlib import Path

# Import helper functions
from portraits.generators.video_helpers import (
    _validate_video_inputs,
    _setup_video_generation_params,
    _prepare_video_generation_environment,
    _generate_video_frames,
    _save_video_file,
    _print_video_generation_summary,
)


def load_pipeline(model_id: str, device: str, dtype, enable_offload: bool = False):
    """Load the SkyReels-V2 video generation pipeline."""
    try:
        from diffusers import DiffusionPipeline
    except ImportError:
        raise ImportError(
            "diffusers is required for video generation. Install with: uv sync --extra video"
        )

    # Handle local model paths that don't exist
    if model_id.startswith("./"):
        local_path = model_id[2:]  # Remove "./"
        if not Path(local_path).exists():
            raise RuntimeError(
                f"Local model not found: {local_path}. Please download the SkyReels-V2 model or use a public model."
            )

    try:
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, variant="fp16")

        if enable_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to(device)

        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load pipeline: {e}")


def generate_video(
    prompt: str,
    model_id: str = None,
    num_frames: int = None,
    width: int = None,
    height: int = None,
    fps: int = None,
    guidance_scale: float = None,
    num_inference_steps: int = None,
    enable_offload: bool = None,
    output_dir: str = None,
) -> str:
    """
    Generate a video from a text prompt using SkyReels-V2 (refactored version).

    This function has been refactored to reduce complexity from 18 to ~8
    by extracting helper functions for each major step.

    Args:
        prompt: Text description of video to generate
        model_id: Local path to model or HuggingFace ID
        num_frames: Number of frames to generate
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second for output video
        guidance_scale: CFG scale for prompt adherence
        num_inference_steps: Number of denoising steps
        enable_offload: Enable CPU offloading to reduce VRAM
        output_dir: Directory to save generated videos

    Returns:
        Path to generated video file
    """
    # Step 1: Validate inputs
    _validate_video_inputs(
        prompt,
        num_frames or 97,
        width or 960,
        height or 544,
        fps or 24,
        guidance_scale or 6.0,
        num_inference_steps or 50,
    )

    # Step 2: Setup parameters with config defaults
    params = _setup_video_generation_params(
        model_id,
        num_frames,
        width,
        height,
        fps,
        guidance_scale,
        num_inference_steps,
        enable_offload,
        output_dir,
    )

    # Step 3: Prepare environment
    output_path, device, dtype = _prepare_video_generation_environment(params["output_dir"])

    # Step 4: Load pipeline
    pipe = load_pipeline(params["model_id"], device, dtype, params["enable_offload"])

    # Step 5: Print generation info
    print(f"\n{'=' * 60}")
    print("Generating video...")
    print(f"Prompt: {prompt}")
    print(f"Resolution: {params['width']}x{params['height']}")
    print(
        f"Frames: {params['num_frames']} ({params['num_frames'] / params['fps']:.1f}s at {params['fps']}fps)"
    )
    print(f"Inference steps: {params['num_inference_steps']}")
    print(f"Guidance scale: {params['guidance_scale']}")
    print(f"{'=' * 60}\n")

    # Step 6: Generate video frames
    frames = _generate_video_frames(
        pipe,
        prompt,
        params["height"],
        params["width"],
        params["num_frames"],
        params["guidance_scale"],
        params["num_inference_steps"],
        device,
    )

    # Step 7: Save video file
    video_path = _save_video_file(
        frames, output_path, params["fps"], params["width"], params["height"]
    )

    # Step 8: Print summary
    _print_video_generation_summary(
        prompt,
        params["num_frames"],
        params["fps"],
        params["width"],
        params["height"],
        params["guidance_scale"],
        params["num_inference_steps"],
        video_path,
    )

    return video_path
