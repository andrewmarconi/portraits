"""Image generation using SDXL Turbo with LoRA support."""

import os
import time
from pathlib import Path

import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from PIL import Image

from ..core.device import get_device_and_dtype
from ..core.exceptions import (
    ConfigurationError,
    DeviceError,
    GenerationError,
    ModelNotFoundError,
    handle_model_load_error,
)
from ..core.utils import ensure_output_dir, get_output_filename, get_timestamp


def generate_headshot(
    prompt: str,
    num_images: int = 1,
    width: int = 512,
    height: int = 512,
    lora_path: str | None = None,
    lora_scale: float = 1.0,
    clip_skip: int = 1,
    guidance_scale: float = 0.0,
    output_dir: str | Path = "output",
) -> str:
    """Generate images using SDXL Turbo with optional LoRA weights.

    Args:
        prompt: Text prompt for generation
        num_images: Number of images to generate
        width: Image width in pixels
        height: Image height in pixels
        lora_path: Path to LoRA weights (.safetensors file or HuggingFace repo)
        lora_scale: LoRA adapter strength (0.0-1.0)
        clip_skip: Number of CLIP layers to skip
        guidance_scale: CFG scale (-100 disables guidance for Turbo)
        output_dir: Output directory for generated images

    Returns:
        Path to generated images directory

    Raises:
        ConfigurationError: If configuration is invalid
        DeviceError: If device selection fails
        GenerationError: If generation fails
        ModelNotFoundError: If model is not found
    """
    try:
        # Validate inputs
        if not prompt or not prompt.strip():
            raise ConfigurationError("Prompt cannot be empty")

        if num_images < 1:
            raise ConfigurationError("num_images must be >= 1")

        if not (0.0 <= lora_scale <= 1.0):
            raise ConfigurationError("lora_scale must be between 0.0 and 1.0")

        if width <= 0 or height <= 0:
            raise ConfigurationError("Width and height must be positive")

        # Get device and dtype
        device, dtype = get_device_and_dtype()

        print(f"🎨 Generating {num_images} image(s) with SDXL Turbo")
        print(f"   Device: {device.upper()} ({dtype})")
        print(f"   Dimensions: {width}x{height}")
        print(f"   Prompt: {prompt}")

        if lora_path:
            print(f"   LoRA: {lora_path} (scale: {lora_scale})")

        # Load pipeline
        pipe = _load_sdxl_pipeline(device, dtype)

        # Load LoRA weights if specified
        if lora_path:
            _load_lora_weights(pipe, lora_path, lora_scale)

        # Ensure output directory exists
        output_path = ensure_output_dir(output_dir)

        # Generate images
        generated_paths = []
        start_time = time.time()

        for i in range(num_images):
            print(f"   Generating image {i + 1}/{num_images}...")

            try:
                image = _generate_single_image(
                    pipe, prompt, width, height, clip_skip, guidance_scale, device
                )

                # Save image
                timestamp = get_timestamp()
                filename = get_output_filename("headshot", "png", i + 1, timestamp)
                image_path = output_path / filename

                image.save(image_path)
                generated_paths.append(str(image_path))

                print(f"     Saved: {image_path.name}")

            except Exception as e:
                raise GenerationError(f"Failed to generate image {i + 1}: {e}")

        # Report results
        elapsed = time.time() - start_time
        print(f"\n✓ Generated {len(generated_paths)} image(s) in {elapsed:.1f}s")
        print(f"📁 Output directory: {output_path}")

        return str(output_path)

    except Exception as e:
        if isinstance(e, (ConfigurationError, DeviceError, GenerationError, ModelNotFoundError)):
            raise
        else:
            raise GenerationError(f"Unexpected error during image generation: {e}")


def _load_sdxl_pipeline(device: str, dtype: torch.dtype) -> AutoPipelineForText2Image:
    """Load SDXL Turbo pipeline with error handling."""
    try:
        print("   Loading SDXL Turbo pipeline...")

        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
        )

        # Move to device
        pipe = pipe.to(device)

        # Optimize for memory
        if device == "cuda":
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()

        print(f"   ✓ Pipeline loaded on {device}")
        return pipe

    except Exception as e:
        error_msg = handle_model_load_error(e, "stabilityai/sdxl-turbo")
        raise ModelNotFoundError(error_msg)


def _load_lora_weights(pipe: AutoPipelineForText2Image, lora_path: str, lora_scale: float) -> None:
    """Load LoRA weights with error handling."""
    try:
        print(f"   Loading LoRA weights: {lora_path}")

        # Check if it's a local file or HuggingFace repo
        if os.path.exists(lora_path):
            if not lora_path.endswith(".safetensors"):
                raise ConfigurationError("LoRA file must be a .safetensors file")
            pipe.load_lora_weights(lora_path, adapter_name="custom")
        else:
            # Assume it's a HuggingFace repo
            pipe.load_lora_weights(lora_path, adapter_name="custom")

        # Set adapter weight
        pipe.set_adapters(["custom"], adapter_weights=[lora_scale])

        print(f"   ✓ LoRA loaded with scale {lora_scale}")

    except Exception as e:
        raise ConfigurationError(f"Failed to load LoRA weights: {e}")


def _generate_single_image(
    pipe: AutoPipelineForText2Image,
    prompt: str,
    width: int,
    height: int,
    clip_skip: int,
    guidance_scale: float,
    device: torch.device,
) -> "Image.Image":
    """Generate a single image."""
    try:
        # SDXL Turbo uses 2 inference steps and no guidance
        num_inference_steps = 2
        guidance_scale = -100.0  # Disable guidance for Turbo

        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            clip_skip=clip_skip,
        ).images[0]

        return image

    except Exception as e:
        raise GenerationError(f"Failed to generate image: {e}")


# Legacy compatibility
def generate_headshot_legacy(*args, **kwargs):
    """Legacy function name for backward compatibility."""
    return generate_headshot(*args, **kwargs)
