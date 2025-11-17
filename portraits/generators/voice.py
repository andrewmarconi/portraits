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
        from snac import SNAC
    except ImportError:
        raise ImportError(
            "transformers and snac are required for voice generation. Install with: uv sync --extra voice"
        )

    try:
        # Use correct model names from config
        model_id = "maya-research/maya1"
        snac_id = "hubertsiuzdak/snac_24khz"

        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, trust_remote_code=True
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        snac_model = SNAC.from_pretrained(snac_id).to(device)

        return model, tokenizer, snac_model
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}")


# Maya1 token constants
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7

SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
BOS_ID = 128000
TEXT_EOT_ID = 128009


def build_prompt_ids(text: str, voice_description: str, tokenizer):
    """Build formatted prompt for Maya1."""
    soh_token = tokenizer.decode([SOH_ID])
    eoh_token = tokenizer.decode([EOH_ID])
    soa_token = tokenizer.decode([SOA_ID])
    sos_token = tokenizer.decode([CODE_START_TOKEN_ID])
    eot_token = tokenizer.decode([TEXT_EOT_ID])
    bos_token = tokenizer.bos_token

    formatted_text = f'<description="{voice_description}"> {text}'

    prompt = soh_token + bos_token + formatted_text + eot_token + eoh_token + soa_token + sos_token

    return tokenizer.encode(prompt, return_tensors="pt")


def extract_snac_codes(token_ids: list) -> list:
    """Extract SNAC codes from generated tokens."""
    try:
        eos_idx = token_ids.index(CODE_END_TOKEN_ID)
    except ValueError:
        eos_idx = len(token_ids)

    snac_codes = [
        token_id for token_id in token_ids[:eos_idx] if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID
    ]

    return snac_codes


def unpack_snac_from_7(snac_tokens: list) -> list:
    """Unpack 7-token SNAC frames to 3 hierarchical levels."""
    if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:
        snac_tokens = snac_tokens[:-1]

    frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
    snac_tokens = snac_tokens[: frames * SNAC_TOKENS_PER_FRAME]

    if frames == 0:
        return [[], [], []]

    l1, l2, l3 = [], [], []

    for i in range(frames):
        slots = snac_tokens[i * 7 : (i + 1) * 7]
        l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend(
            [
                (slots[1] - CODE_TOKEN_OFFSET) % 4096,
                (slots[4] - CODE_TOKEN_OFFSET) % 4096,
            ]
        )
        l3.extend(
            [
                (slots[2] - CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - CODE_TOKEN_OFFSET) % 4096,
            ]
        )

    return [l1, l2, l3]


def generate_voice(
    text: str,
    description: str = "30-year-old, neutral, medium pitch, clear",
    temperature: float = 0.4,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    output_dir: str = "output",
    filename: str | None = None,
    seed: int | None = None,
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
    input_ids = build_prompt_ids(text, params["description"], tokenizer).to(device)

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
