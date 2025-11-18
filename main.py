#!/usr/bin/env python3
"""Unified CLI entry point for Portraits multimodal generation."""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="portraits",
        description="Portraits: Multimodal AI Generation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
   # Generate images
   %(prog)s image --prompt "professional headshot of a woman"
   %(prog)s image --prompt "portrait" --num-images 10 --lora-path ./my_lora.safetensors
   %(prog)s image --prompts-file prompts.txt --num-images 1  # Batch from file
 
   # Generate videos
   %(prog)s video --prompt "cinematic drone shot over mountains"
   %(prog)s video --prompt "underwater coral reef" --num-frames 145
 
   # Generate voice
   %(prog)s voice --text "Hello world!" --voice "warm, friendly, medium pitch"
   %(prog)s voice --text "Welcome to the show! <excited>" --voice "energetic, high pitch"
 
   # Create morphing video
   %(prog)s morph --input output/ --output morphed.mp4
   %(prog)s morph --input images/ --output result.mp4 --fps 30 --morph-frames 12
 
   # Expand prompt templates
   %(prog)s prompts -t "A drag queen with {hair_color} hair."
   %(prog)s prompts -t "{styles} portrait of {ages} person" --sample 10
   %(prog)s prompts --list-variables  # Show AI generation info
 
   # Launch web UI
   %(prog)s ui
   %(prog)s ui --share  # Create public link
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Generation mode")

    # ========== Image Generation ==========
    img_parser = subparsers.add_parser(
        "image",
        help="Generate images using SDXL Turbo",
        description="Generate professional images using SDXL Turbo with optional LoRA weights",
    )
    img_group = img_parser.add_mutually_exclusive_group(required=True)
    img_group.add_argument("--prompt", help="Text prompt for image generation")
    img_group.add_argument("--prompts-file", help="File containing prompts (one per line)")
    img_parser.add_argument(
        "--num-images", type=int, default=1, help="Number of images to generate (default: 1)"
    )
    img_parser.add_argument(
        "--width", type=int, default=512, help="Image width in pixels (default: 512)"
    )
    img_parser.add_argument(
        "--height", type=int, default=512, help="Image height in pixels (default: 512)"
    )
    img_parser.add_argument(
        "--lora-path", help="Path to LoRA weights (.safetensors file or HuggingFace repo)"
    )
    img_parser.add_argument(
        "--lora-scale", type=float, default=1.0, help="LoRA adapter strength 0.0-1.0 (default: 1.0)"
    )
    img_parser.add_argument(
        "--output-dir", default="output", help="Output directory (default: output/)"
    )

    # ========== Video Generation ==========
    vid_parser = subparsers.add_parser(
        "video",
        help="Generate videos from text using SkyReels-V2",
        description="Generate videos from text descriptions using SkyReels-V2-T2V-14B-540P",
    )
    vid_parser.add_argument(
        "--prompt", required=True, help="Text description of the video to generate"
    )
    vid_parser.add_argument(
        "--num-frames", type=int, default=97, help="Number of frames (default: 97 for ~4s at 24fps)"
    )
    vid_parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24)")
    vid_parser.add_argument(
        "--guidance-scale",
        type=float,
        default=6.0,
        help="CFG scale for prompt adherence (default: 6.0)",
    )
    vid_parser.add_argument(
        "--output-dir", default="output", help="Output directory (default: output/)"
    )

    # ========== Voice Generation ==========
    voice_parser = subparsers.add_parser(
        "voice",
        help="Generate speech from text using Maya1",
        description="Generate natural speech from text with customizable voice characteristics",
    )
    voice_parser.add_argument(
        "--text",
        required=True,
        help="Text to convert to speech (supports emotion tags like <laugh>, <excited>)",
    )
    voice_parser.add_argument(
        "--voice", help='Voice description (e.g., "warm, friendly, medium pitch")'
    )
    voice_parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature for variation (default: 0.4)",
    )
    voice_parser.add_argument(
        "--output-dir", default="output", help="Output directory (default: output/)"
    )

    # ========== Mesh Morphing ==========
    morph_parser = subparsers.add_parser(
        "morph",
        help="Create facial morphing videos",
        description="Create smooth morphing videos from a sequence of facial images",
    )
    morph_parser.add_argument("--input", required=True, help="Input directory containing images")
    morph_parser.add_argument(
        "--output", required=True, help="Output video file path (e.g., morphed.mp4)"
    )
    morph_parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24)")
    morph_parser.add_argument(
        "--morph-frames",
        type=int,
        default=8,
        help="Number of interpolation frames between images (default: 8)",
    )
    morph_parser.add_argument(
        "--visualize", action="store_true", help="Draw facial landmarks on output (for debugging)"
    )

    # ========== Prompt Templates ==========
    prompt_parser = subparsers.add_parser(
        "prompts",
        help="Expand prompt templates using AI",
        description="Generate multiple prompt variations from templates using Hugging Face models",
    )
    prompt_parser.add_argument(
        "-t", "--template", help="Prompt template with {variable} placeholders"
    )
    prompt_parser.add_argument(
        "--variables",
        nargs="+",
        help="Variable definitions in format: variable=value1,value2,value3",
    )
    prompt_parser.add_argument(
        "--max-combinations", type=int, help="Maximum number of prompts to generate"
    )
    prompt_parser.add_argument(
        "--sample", type=int, help="Generate sample prompts instead of all combinations"
    )
    prompt_parser.add_argument("-o", "--output", help="Save prompts to file")
    prompt_parser.add_argument(
        "--list-variables", action="store_true", help="Show AI generation information"
    )
    prompt_parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face model for variable generation (default: Qwen2.5-1.5B-Instruct)",
    )

    # ========== Web UI ==========
    ui_parser = subparsers.add_parser(
        "ui",
        help="Launch Gradio web interface",
        description="Launch interactive web interface for all generation features",
    )
    ui_parser.add_argument(
        "--share", action="store_true", help="Create public share link (requires internet)"
    )
    ui_parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command specified
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Execute command
    try:
        if args.command == "image":
            from portraits.generators.image import generate_headshot

            # Handle prompts from file or single prompt
            if args.prompts_file:
                # Read prompts from file
                with open(args.prompts_file, "r", encoding="utf-8") as f:
                    prompts = [line.strip() for line in f if line.strip()]

                print(f"📝 Read {len(prompts)} prompts from {args.prompts_file}")

                # Generate images for each prompt
                for i, prompt in enumerate(prompts, 1):
                    print(f"🎨 Generating image {i}/{len(prompts)}: {prompt}")
                    result = generate_headshot(
                        prompt=prompt,
                        num_images=1,  # One image per prompt
                        width=args.width,
                        height=args.height,
                        lora_path=args.lora_path,
                        lora_scale=args.lora_scale,
                        output_dir=args.output_dir,
                    )
                    print(f"✓ Generated: {result}")

                print(f"\n✓ Generated {len(prompts)} image(s) from prompts file")
            else:
                # Single prompt mode
                result = generate_headshot(
                    prompt=args.prompt,
                    num_images=args.num_images,
                    width=args.width,
                    height=args.height,
                    lora_path=args.lora_path,
                    lora_scale=args.lora_scale,
                    output_dir=args.output_dir,
                )
                print(f"\n✓ Generated {args.num_images} image(s)")

        elif args.command == "video":
            from portraits.generators.video import generate_video

            result = generate_video(
                prompt=args.prompt,
                num_frames=args.num_frames,
                fps=args.fps,
                guidance_scale=args.guidance_scale,
                output_dir=args.output_dir,
            )
            print(f"\n✓ Generated video: {result}")

        elif args.command == "voice":
            from portraits.generators.voice import generate_voice

            result = generate_voice(
                text=args.text,
                description=args.voice,
                temperature=args.temperature,
                output_dir=args.output_dir,
            )
            print(f"\n✓ Generated audio: {result}")

        elif args.command == "morph":
            from portraits.generators.morph import create_mesh_morphing_video

            result = create_mesh_morphing_video(
                input_dir=args.input,
                output_file=args.output,
                fps=args.fps,
                morph_frames=args.morph_frames,
                visualize_landmarks=args.visualize,
            )
            print(f"\n✓ Generated morphing video: {result}")

        elif args.command == "prompts":
            from portraits.generators.smart_prompts import create_smart_template

            # Handle special commands
            if args.list_variables:
                print("\n🤖 LLM-Powered Variable Generation")
                print("=" * 50)
                print("This tool uses Hugging Face models to generate creative variable values!")
                print("\nFeatures:")
                print("• Automatically detects variables in your template")
                print("• Generates context-aware values using Hugging Face models")
                print("• Supports custom variables for specific needs")
                print("• Fallback to rule-based generation if no model available")
                print("\nVariable Examples:")
                print("• {hair_color} → blonde, rainbow, neon pink, platinum...")
                print("• {animals} → cat, dragon, phoenix, robot, alien...")
                print("• {styles} → photorealistic, anime, cyberpunk, oil painting...")
                print("• {emotions} → joyful, mysterious, energetic, peaceful...")
                print("\nJust use descriptive variable names in {brackets}!")
                return

            # Parse custom variables if provided
            custom_variables = {}
            if args.variables:
                for var_def in args.variables:
                    if "=" not in var_def:
                        raise ValueError(
                            f"Invalid variable definition: {var_def}. Use format: variable=value1,value2,value3"
                        )

                    var_name, values_str = var_def.split("=", 1)
                    values = [v.strip() for v in values_str.split(",")]
                    custom_variables[var_name.strip()] = values

            # Create smart template with LLM generation
            print("🧠 Analyzing template and generating variables...")
            prompt_template = create_smart_template(
                args.template,
                custom_variables=custom_variables,
                model_name=args.model,
            )

            print(f"📝 Template: {args.template}")
            print(f"📊 Total combinations: {prompt_template.get_total_combinations()}")

            # Show generated variables
            print(f"\n🎯 Generated variables:")
            for var_name, values in prompt_template.variables.items():
                print(f"  {var_name}: {', '.join(values[:5])}{'...' if len(values) > 5 else ''}")

            # Generate prompts
            if args.sample:
                prompts = prompt_template.generate_prompts(max_combinations=args.sample)
                print(f"\n🎲 Generated {len(prompts)} sample prompts:")
            else:
                prompts = prompt_template.generate_prompts(max_combinations=args.max_combinations)
                print(f"\n📝 Generated {len(prompts)} prompts:")

            # Display prompts
            for i, prompt in enumerate(prompts, 1):
                print(f"{i:2d}. {prompt}")

            # Save to file if requested
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    for i, prompt in enumerate(prompts, 1):
                        f.write(f"{i}. {prompt}\n")
                print(f"\n💾 Saved to {args.output}")

            print(f"\n✓ Generated {len(prompts)} prompt variations")

        elif args.command == "ui":
            from portraits.ui.app import create_app

            print("Launching Gradio web interface...")
            app = create_app()
            app.launch(share=args.share, server_port=args.port, show_error=True)

    except ImportError as e:
        print(f"\n❌ Error: Missing dependencies for '{args.command}' command")
        print(f"   {e}")
        print(f"\nInstall with: uv sync --extra {args.command}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
