#!/usr/bin/env python3
"""
Voice generation helper functions for complexity reduction.

This module contains modular helper functions extracted from the main
generate_voice() function to reduce complexity and improve maintainability.
"""

import torch
import numpy as np
from datetime import datetime

from portraits.core.config import config
from portraits.core.device import get_device_and_dtype
from portraits.core.utils import ensure_output_dir


def _validate_voice_inputs(
    text: str, description: str, temperature: float, top_p: float, max_tokens: int
) -> None:
    """
    Validate voice generation inputs.

    Args:
        text: Text to convert to speech
        description: Voice description
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        max_tokens: Maximum tokens to generate

    Raises:
        ValueError: If inputs are invalid
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    if not description or not description.strip():
        raise ValueError("Description cannot be empty")

    if not 0.0 <= temperature <= 1.0:
        raise ValueError("Temperature must be between 0.0 and 1.0")

    if not 0.0 <= top_p <= 1.0:
        raise ValueError("Top-p must be between 0.0 and 1.0")

    if max_tokens <= 0:
        raise ValueError("Max tokens must be positive")


def _setup_voice_generation_params(
    description: str, temperature: float, top_p: float, max_tokens: int, output_dir: str
) -> dict:
    """
    Setup and validate voice generation parameters with config defaults.

    Args:
        description: Voice description
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        max_tokens: Maximum tokens to generate
        output_dir: Output directory

    Returns:
        Dictionary with validated parameters
    """
    # Set defaults from config if not provided
    if description is None:
        description = config.get(
            "voice_generation.default_voice", "30-year-old, neutral, medium pitch, clear"
        )
    if temperature is None:
        temperature = config.get("voice_generation.temperature", 0.4)
    if top_p is None:
        top_p = config.get("voice_generation.top_p", 0.9)
    if max_tokens is None:
        max_tokens = config.get("voice_generation.max_tokens", 2048)
    if output_dir is None:
        output_dir = config.get("paths.output_dir", "output")

    return {
        "description": description,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "output_dir": output_dir,
    }


def _prepare_voice_generation_environment(output_dir: str) -> tuple:
    """
    Prepare environment for voice generation.

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


def _generate_audio_tokens(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None = None,
) -> list[int]:
    """
    Generate audio tokens using the Maya1 model.

    Args:
        model: Maya1 model
        tokenizer: Maya1 tokenizer
        input_ids: Input token IDs
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        max_tokens: Maximum tokens to generate
        seed: Random seed for reproducibility

    Returns:
        List of generated token IDs

    Raises:
        RuntimeError: If generation fails
    """
    # Import Maya1 constants
    from portraits.generators.voice import CODE_END_TOKEN_ID

    # Set random seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Debug - Using seed: {seed}")

    print("Generating audio tokens...")
    print("Debug - Generation parameters:")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    print(f"  Max new tokens: {max_tokens}")
    print("  Min new tokens: 28 (at least 4 SNAC frames)")
    print("  Repetition penalty: 1.1")
    print()

    try:
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                min_new_tokens=28,  # At least 4 SNAC frames
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                do_sample=True if temperature > 0 else False,
                eos_token_id=CODE_END_TOKEN_ID,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[0, input_ids.shape[1] :].tolist()
        print(f"✓ Generated {len(generated_ids)} tokens")
        return generated_ids

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise RuntimeError(f"Token generation failed: {e}")


def _process_snac_codes(generated_ids: list[int]) -> tuple[list[list[int]], list[int]]:
    """
    Process generated tokens to extract and unpack SNAC codes.

    Args:
        generated_ids: List of generated token IDs

    Returns:
        Tuple of (levels, snac_tokens)

    Raises:
        RuntimeError: If no valid SNAC tokens found
    """
    # Import here to avoid circular imports
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Import the functions from the main voice module
    from portraits.generators.voice import extract_snac_codes, unpack_snac_from_7

    print("Processing audio codes...")

    snac_tokens = extract_snac_codes(generated_ids)

    if not snac_tokens:
        raise RuntimeError("No valid SNAC tokens generated. Try adjusting temperature or text.")

    print(f"✓ Extracted {len(snac_tokens)} SNAC codes")

    levels = unpack_snac_from_7(snac_tokens)
    print(f"✓ Unpacked to 3 hierarchical levels: {[len(level) for level in levels]}")

    return levels, snac_tokens


def _decode_audio_from_codes(
    levels: list[list[int]], device: str, snac_model
) -> tuple[np.ndarray, float]:
    """
    Decode SNAC codes to audio waveform.

    Args:
        levels: Hierarchical SNAC code levels
        device: Device for computation
        snac_model: SNAC decoder model

    Returns:
        Tuple of (audio_array, duration_seconds)

    Raises:
        RuntimeError: If decoding fails
    """
    import numpy as np

    print("Decoding to audio waveform...")
    try:
        codes_tensor = [
            torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0) for level in levels
        ]

        with torch.inference_mode():
            z_q = snac_model.quantizer.from_codes(codes_tensor)
            audio = snac_model.decoder(z_q)[0, 0].cpu().numpy()

        duration = len(audio) / 24000  # 24 kHz sample rate
        print(f"✓ Generated {duration:.2f} seconds of audio")
        return audio, duration

    except Exception as e:
        print(f"❌ Error decoding audio: {e}")
        raise RuntimeError(f"Audio decoding failed: {e}")


def _save_audio_file(audio: np.ndarray, output_path, filename: str | None = None) -> str:
    """
    Save audio array to file.

    Args:
        audio: Audio waveform array
        output_path: Output directory path
        filename: Custom filename (optional)

    Returns:
        Path to saved audio file
    """
    import numpy as np

    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"voice_{timestamp}.wav"

    audio_path = output_path / filename

    try:
        import soundfile as sf

        sf.write(str(audio_path), audio, 24000)
        print(f"✓ Audio saved: {audio_path}")
    except ImportError:
        # Fallback to scipy if soundfile not available
        from scipy.io import wavfile

        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(str(audio_path), 24000, audio_int16)
        print(f"✓ Audio saved (using scipy): {audio_path}")

    return str(audio_path)


def _print_voice_generation_summary(
    text: str, description: str, duration: float, audio_path: str, temperature: float, top_p: float
) -> None:
    """
    Print summary of voice generation results.

    Args:
        text: Input text
        description: Voice description
        duration: Audio duration in seconds
        audio_path: Path to saved audio file
        temperature: Sampling temperature used
        top_p: Top-p sampling used
    """
    print(f"\n{'=' * 60}")
    print("✓ Voice generation complete!")
    print(f"Text: {text}")
    print(f"Voice: {description}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"File: {audio_path}")
    print(f"Settings: Temperature={temperature}, Top-p={top_p}")
    print(f"{'=' * 60}\n")
