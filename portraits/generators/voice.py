#!/usr/bin/env python3
"""
Refactored voice generation with reduced complexity.

This module contains the simplified generate_voice() function that uses
helper functions to reduce complexity from 18 to ~8.
"""

import torch
from datetime import datetime

# Import helper functions
from portraits.generators.voice_helpers import (
    _validate_voice_inputs,
    _setup_voice_generation_params,
    _prepare_voice_generation_environment,
    _generate_audio_tokens,
    _process_snac_codes,
    _decode_audio_from_codes,
    _save_audio_file,
    _print_voice_generation_summary,
)


def load_models(device: str, dtype):
    """Load Maya voice generation models."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import snac
    except ImportError:
        raise ImportError(
            "transformers and snac are required for voice generation. Install with: uv sync --extra voice"
        )

    try:
        model_id = "mayaa-ai/Maya1"

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        snac_model = snac.SNAC.from_pretrained("mayaa-ai/Maya1_SNAC").to(device)

        return model, tokenizer, snac_model
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}")


def build_prompt_ids(text: str, voice_description: str, tokenizer):
    """Build prompt IDs for voice generation."""
    # Simple implementation - combine text and voice description
    prompt = f"{voice_description}: {text}"
    return tokenizer.encode(prompt, return_tensors="pt")


def generate_voice(
    text: str,
    description: str = "30-year-old, neutral, medium pitch, clear",
    temperature: float = 0.4,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    output_dir: str = "output",
    filename: str = None,
    seed: int = None,
) -> str:
    """
    Generate speech from text using Maya1 (refactored version).

    This function has been refactored to reduce complexity from 18 to ~8
    by extracting helper functions for each major step.

    Args:
        text: Text to convert to speech
        description: Natural language voice description
        temperature: Sampling temperature (0.0-1.0, lower = more deterministic)
        top_p: Nucleus sampling threshold (0.0-1.0)
        max_tokens: Maximum number of tokens to generate
        output_dir: Directory to save generated audio
        filename: Custom filename (optional, auto-generated if None)
        seed: Random seed for reproducibility (optional)

    Returns:
        Path to generated audio file
    """
    # Step 1: Validate inputs
    _validate_voice_inputs(text, description, temperature, top_p, max_tokens)

    # Step 2: Setup parameters with config defaults
    params = _setup_voice_generation_params(description, temperature, top_p, max_tokens, output_dir)

    # Step 3: Prepare environment
    output_path, device, dtype = _prepare_voice_generation_environment(params["output_dir"])

    # Step 4: Load models
    model, tokenizer, snac_model = load_models(device, dtype)

    # Step 5: Print generation info
    print(f"\n{'=' * 60}")
    print("Generating speech...")
    print(f"Text: {text}")
    print(f"Voice: {params['description']}")
    print(f"Temperature: {params['temperature']}, Top-p: {params['top_p']}")
    print(f"{'=' * 60}\n")

    # Step 6: Build prompt
    input_ids = build_prompt_ids(tokenizer, params["description"], text).to(device)

    # Step 7: Verify text is in prompt
    decoded_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    if text in decoded_prompt:
        print("✓ Input text verified in prompt\n")
    else:
        print("⚠ WARNING: Input text may not be in prompt!")
        print(f"Debug - Looking for: {repr(text)}\n")

    # Step 8: Generate audio tokens
    generated_ids = _generate_audio_tokens(
        model,
        tokenizer,
        input_ids,
        params["temperature"],
        params["top_p"],
        params["max_tokens"],
        seed,
    )

    # Step 9: Process SNAC codes
    levels, snac_tokens = _process_snac_codes(generated_ids)

    # Step 10: Decode to audio
    audio, duration = _decode_audio_from_codes(levels, device, snac_model)

    # Step 11: Save audio file
    audio_path = _save_audio_file(audio, output_path, filename)

    # Step 12: Print summary
    _print_voice_generation_summary(
        text, params["description"], duration, audio_path, params["temperature"], params["top_p"]
    )

    return audio_path
