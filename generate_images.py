#!/usr/bin/env python3
"""
Generate professional headshots using stabilityai/sdxl-turbo via Hugging Face.
Requires uv for package management.
"""

import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.utils import logging
import os
from datetime import datetime

# Set up logging
logging.set_verbosity_error()

def generate_headshot(prompt="", output_dir="output", width=512, height=512, num_images=1, lora_path=None, lora_scale=1.0, clip_skip=1, guidance_scale=-100):
    """
    Generate a professional headshot using SDXL Turbo.
    
    Args:
        prompt (str): Text prompt for image generation
        output_dir (str): Directory to save generated images
        width (int): Image width (default: 512)
        height (int): Image height (default: 512)
        num_images (int): Number of images to generate (default: 1)
        lora_path (str): Path to LoRA weights (local path or HuggingFace repo)
        lora_scale (float): LoRA adapter scale (default: 1.0)
        clip_skip (int): Number of CLIP layers to skip (default: 1)
        guidance_scale (float): Guidance scale for generation (default: -100)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the pipeline
    print("Loading SDXL Turbo model...")
    
    # Optimize dtype based on device
    if torch.cuda.is_available():
        dtype = torch.bfloat16  # Better for modern GPUs
    elif torch.backends.mps.is_available():
        dtype = torch.float16   # Best for Apple Silicon
    else:
        dtype = torch.float16   # Fallback for CPU
    
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        dtype=dtype,
        variant="fp16"
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    pipe = pipe.to(device)
    
    # Load LoRA if specified
    if lora_path:
        try:
            print(f"Loading LoRA from: {lora_path}")
            pipe.load_lora_weights(lora_path, adapter_name="drawing")
            if hasattr(pipe, 'set_adapters'):
                pipe.set_adapters(["drawing"], adapter_weights=[lora_scale])
            print(f"LoRA loaded with scale: {lora_scale}")
        except Exception as e:
            print(f"Loading default_0 was unsuccessful with the following error:")
            print(f"Error: {e}")
            print("Continuing without LoRA...")
            # Continue without LoRA if loading fails
    
    # Quality optimizations
    # Skip torch.compile on hardware with limited SMs to avoid warnings
    
    # Keep VAE in float32 for better quality
    if hasattr(pipe, 'upcast_vae'):
        pipe.upcast_vae()
    
    print(f"Using device: {device}")
    print(f"Generating {num_images} image(s) with prompt: '{prompt}'")
    
    generated_paths = []
    
    for i in range(num_images):
        print(f"Generating image {i+1}/{num_images}...")
        
        # Generate image with quality optimizations
        # Use 2-4 steps for better quality (still very fast)
        image = pipe(
            prompt=prompt,
            num_inference_steps=2,  # Increased from 1 for better quality
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            clip_skip=clip_skip
        ).images[0]
        
        # Save image with timestamp and index
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if num_images > 1:
            filename = f"headshot_{timestamp}_{i+1:02d}.png"
        else:
            filename = f"headshot_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        image.save(filepath)
        generated_paths.append(filepath)
        print(f"Image {i+1} saved to: {filepath}")
    
    print(f"Successfully generated {len(generated_paths)} image(s)")
    return generated_paths if num_images > 1 else generated_paths[0]

if __name__ == "__main__":
    
    # Generate the headshot
    result_path = generate_headshot(
        num_images=64, 
        prompt="a professional headshot of a drag queen, face in center of frame, rave schene, 1990s",
        lora_path="~/Documents/ComfyUI/models/LoRA/RaveSceneNineties.safetensors"
        # lora_scale=0.7,
        # clip_skip=1,
        # guidance_scale=-100
    )
    print(f"Image generated successfully: {result_path}")