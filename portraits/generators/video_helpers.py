#!/usr/bin/env python3
"""
Video generation helper functions for complexity reduction.

This module contains modular helper functions extracted from the main
generate_video() function to reduce complexity and improve maintainability.
"""

import torch
from datetime import datetime

from portraits.core.config import config
from portraits.core.device import get_device_and_dtype
from portraits.core.utils import ensure_output_dir


def _validate_video_inputs(prompt: str, num_frames: int, width: int, height: int, 
                         fps: int, guidance_scale: float, num_inference_steps: int) -> None:
    """
    Validate video generation inputs.
    
    Args:
        prompt: Text description of video
        num_frames: Number of frames to generate
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second
        guidance_scale: CFG scale for prompt adherence
        num_inference_steps: Number of denoising steps
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    if num_frames <= 0:
        raise ValueError("Number of frames must be positive")
    
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    
    if fps <= 0:
        raise ValueError("FPS must be positive")
    
    if guidance_scale <= 0:
        raise ValueError("Guidance scale must be positive")
    
    if num_inference_steps <= 0:
        raise ValueError("Number of inference steps must be positive")


def _setup_video_generation_params(model_id: str, num_frames: int, width: int, height: int,
                                fps: int, guidance_scale: float, num_inference_steps: int,
                                enable_offload: bool, output_dir: str) -> dict:
    """
    Setup and validate video generation parameters with config defaults.
    
    Args:
        model_id: Model ID or path
        num_frames: Number of frames
        width: Video width
        height: Video height
        fps: Frames per second
        guidance_scale: CFG scale
        num_inference_steps: Inference steps
        enable_offload: Enable CPU offloading
        output_dir: Output directory
        
    Returns:
        Dictionary with validated parameters
    """
    # Set defaults from config if not provided
    if model_id is None:
        model_id = config.get("models.skyreels_v2", "./SkyReels-V2-T2V-14B-540P")
    if num_frames is None:
        num_frames = config.get("video_generation.num_frames", 97)
    if width is None:
        width = config.get("video_generation.width", 960)
    if height is None:
        height = config.get("video_generation.height", 544)
    if fps is None:
        fps = config.get("video_generation.fps", 24)
    if guidance_scale is None:
        guidance_scale = config.get("video_generation.guidance_scale", 6.0)
    if num_inference_steps is None:
        num_inference_steps = config.get("video_generation.num_inference_steps", 50)
    if enable_offload is None:
        enable_offload = config.get("video_generation.enable_offload", True)
    if output_dir is None:
        output_dir = config.get("paths.output_dir", "output")

    return {
        "model_id": model_id,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "fps": fps,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "enable_offload": enable_offload,
        "output_dir": output_dir
    }


def _prepare_video_generation_environment(output_dir: str) -> tuple:
    """
    Prepare environment for video generation.
    
    Args:
        output_dir: Output directory path
        
    Returns:
        Tuple of (output_path, device, dtype)
    """
    # Create output directory
    output_path = ensure_output_dir(output_dir)
    
    # Get device and dtype
    device, dtype = get_device_and_dtype()
    
    return output_path, device, dtype


def _generate_video_frames(pipe, prompt: str, height: int, width: int, num_frames: int,
                        guidance_scale: float, num_inference_steps: int, device: str) -> list:
    """
    Generate video frames using the SkyReels-V2 pipeline.
    
    Args:
        pipe: SkyReels-V2 pipeline
        prompt: Text description of video
        height: Video height
        width: Video width
        num_frames: Number of frames
        guidance_scale: CFG scale
        num_inference_steps: Number of inference steps
        device: Device for generation
        
    Returns:
        List of generated frames
        
    Raises:
        RuntimeError: If generation fails
    """
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
        return frames

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
        raise RuntimeError(f"Video generation failed: {e}")
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise RuntimeError(f"Video generation failed: {e}")


def _save_video_file(frames: list, output_path, fps: int, width: int, height: int) -> str:
    """
    Save generated frames to video file.
    
    Args:
        frames: List of video frames
        output_path: Output directory path
        fps: Frames per second
        width: Video width
        height: Video height
        
    Returns:
        Path to saved video file
    """
    import numpy as np
    from PIL import Image
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"video_{timestamp}.mp4"
    video_path = output_path / video_filename

    # Try to use diffusers export utility
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, str(video_path), fps=fps)
        print(f"✓ Video saved using diffusers export: {video_path}")
        return str(video_path)
    except ImportError:
        # Fallback to manual export using opencv
        import cv2

        print("⚠ Using OpenCV fallback for video export")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

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
        return str(video_path)


def _print_video_generation_summary(prompt: str, num_frames: int, fps: int, 
                                width: int, height: int, guidance_scale: float,
                                num_inference_steps: int, video_path: str) -> None:
    """
    Print summary of video generation results.
    
    Args:
        prompt: Input prompt
        num_frames: Number of frames generated
        fps: Frames per second
        width: Video width
        height: Video height
        guidance_scale: CFG scale used
        num_inference_steps: Inference steps used
        video_path: Path to saved video file
    """
    duration = num_frames / fps
    print(f"\n{'='*60}")
    print("✓ Generation complete!")
    print(f"Prompt: {prompt}")
    print(f"Resolution: {width}x{height}")
    print(f"Duration: {duration:.1f} seconds ({num_frames} frames at {fps}fps)")
    print(f"Settings: Guidance={guidance_scale}, Steps={num_inference_steps}")
    print(f"File: {video_path}")
    print(f"{'='*60}\n")