#!/usr/bin/env python3
"""
Text-to-speech generation using maya-research/maya1

This script generates natural-sounding speech from text using the Maya1 model,
a 3B-parameter decoder-only transformer optimized for voice synthesis.

Installation:
    uv add snac soundfile

Usage:
    uv run python generate_voice.py

Requirements:
    - VRAM: 16GB+ recommended (A100, H100, or RTX 4090)
    - Python 3.11+
    - PyTorch with CUDA/MPS support

Device Support:
    - CUDA: Full support, recommended for best performance
    - MPS (Apple Silicon): Supported with float16
    - CPU: Supported but slow

Features:
    - Natural language voice descriptions (age, pitch, emotion, etc.)
    - 20+ emotion tags (<laugh>, <cry>, <whisper>, <angry>, etc.)
    - Multi-accent English support
    - 24 kHz mono audio output
    - Streaming capability (sub-100ms latency with vLLM)
"""

from datetime import datetime

import numpy as np
import torch

from portraits.core.config import config

# Import shared modules
from portraits.core.device import get_device_and_dtype
from portraits.core.utils import ensure_output_dir

# Maya1 Token Constants
SOH_ID = 128259  # Start of Header
EOH_ID = 128260  # End of Header
SOA_ID = 128261  # Start of Audio
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
TEXT_EOT_ID = 128009  # Text End of Transmission
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7
BOS_ID = 128000


def extract_snac_codes(token_ids: list[int]) -> list[int]:
    """
    Extract SNAC audio codes from generated tokens.

    Args:
        token_ids: List of generated token IDs

    Returns:
        List of SNAC audio codes
    """
    try:
        eos_idx = token_ids.index(CODE_END_TOKEN_ID)
    except ValueError:
        eos_idx = len(token_ids)

    snac_codes = [
        token_id for token_id in token_ids[:eos_idx] if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID
    ]

    return snac_codes


def unpack_snac_from_7(snac_tokens: list[int]) -> list[list[int]]:
    """
    Unpack 7-token SNAC frames into 3 hierarchical levels.

    SNAC uses a hierarchical encoding with 3 levels of audio codes.
    Each frame contains 7 tokens that must be unpacked into these levels.

    Args:
        snac_tokens: List of SNAC token IDs

    Returns:
        List containing 3 levels: [level1, level2, level3]
    """
    # Remove end token if present
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
        l2.extend([(slots[1] - CODE_TOKEN_OFFSET) % 4096, (slots[4] - CODE_TOKEN_OFFSET) % 4096])
        l3.extend(
            [
                (slots[2] - CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - CODE_TOKEN_OFFSET) % 4096,
            ]
        )

    return [l1, l2, l3]


def build_prompt_ids(tokenizer, description: str, text: str) -> torch.Tensor:
    """
    Build a properly formatted prompt for Maya1 as token IDs.

    The prompt structure is:
    [SOH] [BOS] <description="..."> [text] [EOT] [EOH] [SOA] [SOS]

    Where:
    - SOH/EOH delimit the header section (text input)
    - The description tag provides voice characteristics
    - SOA/SOS mark the start of audio generation

    Args:
        tokenizer: Maya1 tokenizer
        description: Natural language voice description
                     e.g., "40-year-old, warm, low pitch, conversational"
        text: Text to convert to speech

    Returns:
        Tensor of token IDs ready for generation
    """
    # Format the text content with description metadata
    formatted_text = f'<description="{description}"> {text}'

    # Tokenize just the text content (no special tokens)
    text_tokens = tokenizer.encode(formatted_text, add_special_tokens=False)

    # Build the complete sequence with special tokens as IDs
    prompt_ids = (
        [SOH_ID]  # Start of Header
        + [BOS_ID]  # Begin of Sequence
        + text_tokens  # Text content with description
        + [TEXT_EOT_ID]  # End of Text
        + [EOH_ID]  # End of Header
        + [SOA_ID]  # Start of Audio
        + [CODE_START_TOKEN_ID]  # Start of Code (audio tokens)
    )

    # Debug: Print token structure
    print("Debug - Prompt token sequence:")
    print(f"  Total tokens: {len(prompt_ids)}")
    print(f"  First 10 token IDs: {prompt_ids[:10]}")
    print(f"  Last 5 token IDs: {prompt_ids[-5:]}")
    print(f"  Text portion length: {len(text_tokens)} tokens")

    # Decode to verify what the model will see
    decoded = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    print("\nDebug - Decoded prompt:")
    print(f"  {repr(decoded[:200])}...")

    return torch.tensor([prompt_ids], dtype=torch.long)


def load_models(device: str, dtype: torch.dtype):
    """
    Load Maya1 model and SNAC decoder.

    Args:
        device: Target device (cuda/mps/cpu)
        dtype: Model dtype (bfloat16/float16)

    Returns:
        Tuple of (maya_model, tokenizer, snac_model)
    """
    print("\nLoading Maya1 model...")
    print("⚠ This is a 3B parameter model - initial download may be large (~6GB)")

    try:
        from snac import SNAC
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            f"Required library not found: {e}\n"
            "Please install dependencies:\n"
            "  uv add snac soundfile\n"
            "  uv sync"
        )

    # Load Maya1 model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "maya-research/maya1",
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        if device != "cuda":
            model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained("maya-research/maya1", trust_remote_code=True)

        print("✓ Maya1 model loaded successfully")

    except Exception as e:
        print(f"❌ Error loading Maya1: {e}")
        raise

    # Load SNAC decoder
    print("\nLoading SNAC audio decoder...")
    try:
        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        snac_model = snac_model.to(device)
        print("✓ SNAC decoder loaded successfully")
    except Exception as e:
        print(f"❌ Error loading SNAC decoder: {e}")
        raise

    return model, tokenizer, snac_model


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
    Generate speech from text using Maya1.

    Args:
        text: Text to convert to speech
        description: Natural language voice description
                     Examples:
                     - "40-year-old, warm, low pitch, conversational"
                     - "young female, energetic, high pitch"
                     - "elderly male, calm, deep voice"
        temperature: Sampling temperature (0.0-1.0, lower = more deterministic)
        top_p: Nucleus sampling threshold (0.0-1.0)
        max_tokens: Maximum number of tokens to generate
        output_dir: Directory to save generated audio
        filename: Custom filename (optional, auto-generated if None)
        seed: Random seed for reproducibility (optional)

    Returns:
        Path to generated audio file
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

    # Create output directory
    output_path = ensure_output_dir(output_dir)

    # Get device and dtype
    device, dtype = get_device_and_dtype()

    # Load models
    model, tokenizer, snac_model = load_models(device, dtype)

    print(f"\n{'='*60}")
    print("Generating speech...")
    print(f"Text: {text}")
    print(f"Voice: {description}")
    print(f"Temperature: {temperature}, Top-p: {top_p}")
    print(f"{'='*60}\n")

    # Build prompt as token IDs directly
    input_ids = build_prompt_ids(tokenizer, description, text).to(device)

    # Verify text is in the prompt
    decoded_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    if text in decoded_prompt:
        print("✓ Input text verified in prompt\n")
    else:
        print("⚠ WARNING: Input text may not be in prompt!")
        print(f"Debug - Looking for: {repr(text)}\n")

    # Set random seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Debug - Using seed: {seed}")

    # Generate SNAC tokens
    print("Generating audio tokens...")
    print("Debug - Generation parameters:")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    print(f"  Max new tokens: {max_tokens}")
    print("  Repetition penalty: 1.1")
    print()

    try:
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                do_sample=True if temperature > 0 else False,
                eos_token_id=CODE_END_TOKEN_ID,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=3,  # Prevent repetitive patterns
            )

        generated_ids = outputs[0, input_ids.shape[1] :].tolist()
        print(f"✓ Generated {len(generated_ids)} tokens")

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise

    # Extract and unpack SNAC codes
    print("Processing audio codes...")
    snac_tokens = extract_snac_codes(generated_ids)

    if not snac_tokens:
        raise RuntimeError("No valid SNAC tokens generated. Try adjusting temperature or text.")

    print(f"✓ Extracted {len(snac_tokens)} SNAC codes")

    levels = unpack_snac_from_7(snac_tokens)
    print(f"✓ Unpacked to 3 hierarchical levels: {[len(level) for level in levels]}")

    # Decode to audio
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

    except Exception as e:
        print(f"❌ Error decoding audio: {e}")
        raise

    # Save audio file
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

        # Normalize to int16 range
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(str(audio_path), 24000, audio_int16)
        print(f"✓ Audio saved (using scipy): {audio_path}")

    print(f"\n{'='*60}")
    print("✓ Voice generation complete!")
    print(f"Duration: {duration:.2f} seconds")
    print(f"File: {audio_path}")
    print(f"{'='*60}\n")

    return str(audio_path)


def main():
    """Main entry point with example usage."""

    # Example voice generations - modify these or add your own
    examples = [
        {
            "text": "Hello! I'm Maya, a text-to-speech system that can generate natural-sounding voices.",
            "description": "30-year-old, friendly, medium pitch, clear",
            "seed": 42,  # Fixed seed for reproducibility
        },
        # {
        #     "text": "The quick brown fox jumps over the lazy dog. <laugh> Just kidding, that's such a cliche!",
        #     "description": "young female, playful, high pitch, energetic",
        #     "seed": 123,
        # },
        # {
        #     "text": "In a world where technology advances rapidly, <whisper> we must remember to pause and reflect.",
        #     "description": "middle-aged male, thoughtful, low pitch, calm",
        #     "seed": 456,
        # },
    ]

    # Generation settings
    settings = {
        "temperature": 0.4,  # Lower = more consistent, higher = more varied
        "top_p": 0.9,  # Nucleus sampling threshold
        "max_tokens": 2048,  # Maximum audio tokens to generate
    }

    print("\n" + "=" * 60)
    print("DIAGNOSTIC MODE ENABLED")
    print("=" * 60)
    print("Debug information will be printed to help troubleshoot")
    print("If the generated speech doesn't match your text:")
    print("  1. Check 'Prompt preview' to verify text is included")
    print("  2. Check '✓ Input text verified' appears")
    print("  3. Try lowering temperature (0.3) for more consistency")
    print("  4. Use a fixed seed for reproducibility")
    print("=" * 60 + "\n")

    print("Maya1 Text-to-Speech Generator")
    print("=" * 60)
    print(f"Generating {len(examples)} voice sample(s)...")
    print()

    # Emotion tags you can use in text:
    print("Available emotion tags:")
    print("  <laugh>, <cry>, <whisper>, <angry>, <gasp>, <sigh>,")
    print("  <excited>, <sad>, <confused>, <surprised>, and more!")
    print()

    # Generate voices for each example
    for i, example in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] Processing...")
        try:
            # Extract seed if provided
            seed = example.get("seed", None)

            audio_path = generate_voice(
                text=example["text"], description=example["description"], seed=seed, **settings
            )
            print(f"✓ Sample {i} saved to: {audio_path}")
        except Exception as e:
            print(f"❌ Failed to generate sample {i}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n✓ All voice samples generated successfully!")


if __name__ == "__main__":
    main()
