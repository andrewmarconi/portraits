#!/usr/bin/env python3
"""Unified CLI entry point for Portraits multimodal generation."""
import argparse
import sys
from pathlib import Path

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='portraits',
        description='Portraits: Multimodal AI Generation Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate images
  %(prog)s image --prompt "professional headshot of a woman"
  %(prog)s image --prompt "portrait" --num-images 10 --lora-path ./my_lora.safetensors

  # Generate videos
  %(prog)s video --prompt "cinematic drone shot over mountains"
  %(prog)s video --prompt "underwater coral reef" --num-frames 145

  # Generate voice
  %(prog)s voice --text "Hello world!" --voice "warm, friendly, medium pitch"
  %(prog)s voice --text "Welcome to the show! <excited>" --voice "energetic, high pitch"

  # Create morphing video
  %(prog)s morph --input output/ --output morphed.mp4
  %(prog)s morph --input images/ --output result.mp4 --fps 30 --morph-frames 12

  # Launch web UI
  %(prog)s ui
  %(prog)s ui --share  # Create public link
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Generation mode')

    # ========== Image Generation ==========
    img_parser = subparsers.add_parser(
        'image',
        help='Generate images using SDXL Turbo',
        description='Generate professional images using SDXL Turbo with optional LoRA weights'
    )
    img_parser.add_argument(
        '--prompt',
        required=True,
        help='Text prompt for image generation'
    )
    img_parser.add_argument(
        '--num-images',
        type=int,
        default=1,
        help='Number of images to generate (default: 1)'
    )
    img_parser.add_argument(
        '--width',
        type=int,
        default=512,
        help='Image width in pixels (default: 512)'
    )
    img_parser.add_argument(
        '--height',
        type=int,
        default=512,
        help='Image height in pixels (default: 512)'
    )
    img_parser.add_argument(
        '--lora-path',
        help='Path to LoRA weights (.safetensors file or HuggingFace repo)'
    )
    img_parser.add_argument(
        '--lora-scale',
        type=float,
        default=1.0,
        help='LoRA adapter strength 0.0-1.0 (default: 1.0)'
    )
    img_parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory (default: output/)'
    )

    # ========== Video Generation ==========
    vid_parser = subparsers.add_parser(
        'video',
        help='Generate videos from text using SkyReels-V2',
        description='Generate videos from text descriptions using SkyReels-V2-T2V-14B-540P'
    )
    vid_parser.add_argument(
        '--prompt',
        required=True,
        help='Text description of the video to generate'
    )
    vid_parser.add_argument(
        '--num-frames',
        type=int,
        default=97,
        help='Number of frames (default: 97 for ~4s at 24fps)'
    )
    vid_parser.add_argument(
        '--fps',
        type=int,
        default=24,
        help='Frames per second (default: 24)'
    )
    vid_parser.add_argument(
        '--guidance-scale',
        type=float,
        default=6.0,
        help='CFG scale for prompt adherence (default: 6.0)'
    )
    vid_parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory (default: output/)'
    )

    # ========== Voice Generation ==========
    voice_parser = subparsers.add_parser(
        'voice',
        help='Generate speech from text using Maya1',
        description='Generate natural speech from text with customizable voice characteristics'
    )
    voice_parser.add_argument(
        '--text',
        required=True,
        help='Text to convert to speech (supports emotion tags like <laugh>, <excited>)'
    )
    voice_parser.add_argument(
        '--voice',
        help='Voice description (e.g., "warm, friendly, medium pitch")'
    )
    voice_parser.add_argument(
        '--temperature',
        type=float,
        default=0.4,
        help='Sampling temperature for variation (default: 0.4)'
    )
    voice_parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory (default: output/)'
    )

    # ========== Mesh Morphing ==========
    morph_parser = subparsers.add_parser(
        'morph',
        help='Create facial morphing videos',
        description='Create smooth morphing videos from a sequence of facial images'
    )
    morph_parser.add_argument(
        '--input',
        required=True,
        help='Input directory containing images'
    )
    morph_parser.add_argument(
        '--output',
        required=True,
        help='Output video file path (e.g., morphed.mp4)'
    )
    morph_parser.add_argument(
        '--fps',
        type=int,
        default=24,
        help='Frames per second (default: 24)'
    )
    morph_parser.add_argument(
        '--morph-frames',
        type=int,
        default=8,
        help='Number of interpolation frames between images (default: 8)'
    )
    morph_parser.add_argument(
        '--visualize',
        action='store_true',
        help='Draw facial landmarks on output (for debugging)'
    )

    # ========== Web UI ==========
    ui_parser = subparsers.add_parser(
        'ui',
        help='Launch Gradio web interface',
        description='Launch interactive web interface for all generation features'
    )
    ui_parser.add_argument(
        '--share',
        action='store_true',
        help='Create public share link (requires internet)'
    )
    ui_parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Server port (default: 7860)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command specified
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Execute command
    try:
        if args.command == 'image':
            from portraits.generators.image import generate_headshot
            result = generate_headshot(
                prompt=args.prompt,
                num_images=args.num_images,
                width=args.width,
                height=args.height,
                lora_path=args.lora_path,
                lora_scale=args.lora_scale,
                output_dir=args.output_dir
            )
            print(f"\n✓ Generated {args.num_images} image(s)")

        elif args.command == 'video':
            from portraits.generators.video import generate_video
            result = generate_video(
                prompt=args.prompt,
                num_frames=args.num_frames,
                fps=args.fps,
                guidance_scale=args.guidance_scale,
                output_dir=args.output_dir
            )
            print(f"\n✓ Generated video: {result}")

        elif args.command == 'voice':
            from portraits.generators.voice import generate_voice
            result = generate_voice(
                text=args.text,
                description=args.voice,
                temperature=args.temperature,
                output_dir=args.output_dir
            )
            print(f"\n✓ Generated audio: {result}")

        elif args.command == 'morph':
            from portraits.generators.morph import create_mesh_morphing_video
            result = create_mesh_morphing_video(
                input_dir=args.input,
                output_file=args.output,
                fps=args.fps,
                morph_frames=args.morph_frames,
                visualize_landmarks=args.visualize
            )
            print(f"\n✓ Generated morphing video: {result}")

        elif args.command == 'ui':
            from portraits.ui.app import create_app
            print("Launching Gradio web interface...")
            app = create_app()
            app.launch(
                share=args.share,
                server_port=args.port,
                show_error=True
            )

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