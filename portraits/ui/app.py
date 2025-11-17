#!/usr/bin/env python3
"""
Multimodal Generative AI Web Interface

A Gradio-based web application that integrates all generative AI capabilities:
- Image Generation (SDXL Turbo + LoRA)
- Text-to-Video (SkyReels-V2)
- Text-to-Speech (Maya1)
- Video Morphing (MediaPipe Face Mesh)

Usage:
    uv run python app.py

This will start a local web server with an interactive interface for all models.
"""

import warnings

import gradio as gr

# Suppress warnings
warnings.filterwarnings("ignore")

# Import generation functions
try:
    from portraits.generators.image import generate_headshot

    HEADSHOT_AVAILABLE = True
except ImportError:
    HEADSHOT_AVAILABLE = False
    print("⚠ Image generation not available")

try:
    from portraits.generators.video import generate_video

    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False
    print("⚠ Text-to-video not available")

try:
    from portraits.generators.voice import generate_voice

    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("⚠ Text-to-speech not available")

try:
    from portraits.generators.morph import create_mesh_morphing_video

    MORPHING_AVAILABLE = True
except ImportError:
    MORPHING_AVAILABLE = False
    print("⚠ Video morphing not available")


# ============================================================================
# Image Generation Tab
# ============================================================================


def generate_image_ui(prompt, num_images, lora_path, lora_scale, clip_skip, width, height):
    """Gradio wrapper for image generation."""
    if not HEADSHOT_AVAILABLE:
        return None, "Image generation not available. Check dependencies."

    try:
        output_dir = generate_headshot(
            prompt=prompt,
            num_images=int(num_images),
            lora_path=lora_path if lora_path.strip() else None,
            lora_scale=float(lora_scale),
            width=int(width),
            height=int(height),
            clip_skip=int(clip_skip),
        )
        
        # Convert directory path to list of generated images for Gradio gallery
        import glob
        from pathlib import Path
        
        # Find all generated images in the output directory
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(str(Path(output_dir) / ext)))
        
        # Sort by creation time (newest first)
        image_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        
        return image_files, f"✓ Generated {len(image_files)} image(s) successfully!"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_image_tab():
    """Create the image generation tab interface."""
    with gr.Tab("🖼️ Image Generation"):
        gr.Markdown(
            """
        # SDXL Turbo Image Generation
        Generate high-quality images with optional LoRA style customization.
        """
        )

        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Prompt", placeholder="A professional headshot of a person...", lines=3
                )

                with gr.Row():
                    num_images_input = gr.Slider(
                        minimum=1, maximum=10, value=1, step=1, label="Number of Images"
                    )

                with gr.Accordion("Advanced Settings", open=False):
                    lora_path_input = gr.Textbox(
                        label="LoRA Path (optional)",
                        placeholder="./Ink_drawing_style-000001.safetensors",
                        value="",
                    )
                    lora_scale_input = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.8, step=0.1, label="LoRA Scale"
                    )
                    clip_skip_input = gr.Slider(
                        minimum=0, maximum=2, value=0, step=1, label="CLIP Skip"
                    )
                    width_input = gr.Slider(
                        minimum=256, maximum=1024, value=512, step=64, label="Width"
                    )
                    height_input = gr.Slider(
                        minimum=256, maximum=1024, value=512, step=64, label="Height"
                    )

                generate_btn = gr.Button("Generate Images", variant="primary")

            with gr.Column():
                image_output = gr.Gallery(label="Generated Images", columns=2, height=500)
                status_output = gr.Textbox(label="Status", lines=2)

        generate_btn.click(
            fn=generate_image_ui,
            inputs=[
                prompt_input,
                num_images_input,
                lora_path_input,
                lora_scale_input,
                clip_skip_input,
                width_input,
                height_input,
            ],
            outputs=[image_output, status_output],
        )


# ============================================================================
# Text-to-Video Tab
# ============================================================================


def generate_video_ui(prompt, num_frames, guidance_scale, num_inference_steps):
    """Gradio wrapper for video generation."""
    if not VIDEO_AVAILABLE:
        return None, "Text-to-video not available. Please download the SkyReels-V2 model."

    try:
        video_path = generate_video(
            prompt=prompt,
            num_frames=int(num_frames),
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
        )

        return video_path, f"✓ Video generated successfully: {video_path}"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_video_tab():
    """Create the text-to-video tab interface."""
    with gr.Tab("🎬 Text-to-Video"):
        gr.Markdown(
            """
        # SkyReels-V2 Video Generation
        Generate videos from text descriptions (requires downloaded model).

        **Note:** Model must be downloaded first:
        ```bash
        git lfs install
        git clone https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-540P
        ```
        """
        )

        with gr.Row():
            with gr.Column():
                video_prompt_input = gr.Textbox(
                    label="Video Prompt",
                    placeholder="A cinematic shot of a mountain landscape at sunrise...",
                    lines=4,
                )

                num_frames_input = gr.Slider(
                    minimum=25,
                    maximum=145,
                    value=97,
                    step=12,
                    label="Number of Frames (~4 sec at 24fps = 97 frames)",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    guidance_scale_input = gr.Slider(
                        minimum=1.0, maximum=15.0, value=6.0, step=0.5, label="Guidance Scale"
                    )
                    num_inference_steps_input = gr.Slider(
                        minimum=20, maximum=100, value=50, step=5, label="Inference Steps"
                    )

                generate_video_btn = gr.Button("Generate Video", variant="primary")

            with gr.Column():
                video_output = gr.Video(label="Generated Video")
                video_status_output = gr.Textbox(label="Status", lines=3)

        generate_video_btn.click(
            fn=generate_video_ui,
            inputs=[
                video_prompt_input,
                num_frames_input,
                guidance_scale_input,
                num_inference_steps_input,
            ],
            outputs=[video_output, video_status_output],
        )


# ============================================================================
# Text-to-Speech Tab
# ============================================================================


def generate_voice_ui(text, description, temperature, top_p):
    """Gradio wrapper for voice generation."""
    if not VOICE_AVAILABLE:
        return None, "Text-to-speech not available. Check dependencies (snac, soundfile)."

    try:
        audio_path = generate_voice(
            text=text, description=description, temperature=float(temperature), top_p=float(top_p)
        )

        return audio_path, f"✓ Voice generated successfully: {audio_path}"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_voice_tab():
    """Create the text-to-speech tab interface."""
    with gr.Tab("🎙️ Text-to-Speech"):
        gr.Markdown(
            """
        # Maya1 Voice Generation
        Generate natural-sounding speech with customizable voice characteristics and emotions.

        **Available Emotion Tags:** `<laugh>`, `<cry>`, `<whisper>`, `<angry>`, `<gasp>`, `<excited>`, `<sad>`, and more!
        """
        )

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text to Speak",
                    placeholder="Hello! I'm an AI voice. <laugh> Pretty cool, right?",
                    lines=5,
                )

                description_input = gr.Textbox(
                    label="Voice Description",
                    placeholder="30-year-old, friendly, medium pitch, clear",
                    value="30-year-old, neutral, medium pitch, clear",
                )

                with gr.Accordion("Voice Examples", open=True):
                    gr.Markdown(
                        """
                    **Preset Voices:**
                    - `40-year-old, warm, low pitch, conversational`
                    - `young female, energetic, high pitch`
                    - `elderly male, calm, deep voice, storytelling`
                    - `middle-aged, professional, medium pitch, confident`
                    """
                    )

                with gr.Accordion("Advanced Settings", open=False):
                    temperature_input = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.4,
                        step=0.1,
                        label="Temperature (lower = more consistent)",
                    )
                    top_p_input = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top P (nucleus sampling)",
                    )

                generate_voice_btn = gr.Button("Generate Voice", variant="primary")

            with gr.Column():
                audio_output = gr.Audio(label="Generated Voice", type="filepath")
                voice_status_output = gr.Textbox(label="Status", lines=3)

        generate_voice_btn.click(
            fn=generate_voice_ui,
            inputs=[text_input, description_input, temperature_input, top_p_input],
            outputs=[audio_output, voice_status_output],
        )


# ============================================================================
# Video Morphing Tab
# ============================================================================


def create_morphing_tab():
    """Create the video morphing tab interface."""
    with gr.Tab("🎭 Video Morphing"):
        gr.Markdown(
            """
        # Face Mesh Morphing
        Create smooth morphing videos from a sequence of images using MediaPipe Face Mesh.

        **Note:** This feature requires uploading multiple face images or using generated images from the output folder.
        """
        )

        gr.Markdown(
            """
        ### How to Use:
        1. Generate multiple images using the Image Generation tab
2. Run morphing from command line:
         ```bash
         uv run python main.py morph --input output/ --output morphed.mp4
         ```

        **Current:** Use CLI command above or integrate with generated images from output folder
        """
        )


# ============================================================================
# Main App
# ============================================================================


def create_app():
    """Create the main Gradio application."""

    with gr.Blocks(title="Multimodal AI Studio", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
        # 🎨 Multimodal AI Generation Studio

        A unified interface for generating images, videos, speech, and morphing effects using state-of-the-art AI models.

        **Available Models:**
        - 🖼️ **SDXL Turbo** - Fast, high-quality image generation with LoRA support
        - 🎬 **SkyReels-V2** - 14B parameter text-to-video model (540P, 24fps)
        - 🎙️ **Maya1** - 3B parameter natural voice synthesis with emotion support
        - 🎭 **MediaPipe** - Professional facial morphing with 468-point mesh tracking
        """
        )

        # Create tabs
        create_image_tab()
        create_video_tab()
        create_voice_tab()
        create_morphing_tab()

        # Footer
        gr.Markdown(
            """
        ---
        **System Info:**
        - Device auto-detection (CUDA/MPS/CPU)
        - All generated content saved to `output/` directory
        - Models downloaded automatically from HuggingFace (except SkyReels-V2)

        **Tips:**
        - Use CUDA GPU for best performance
        - Apple Silicon (MPS) supported for all models
        - CPU mode works but is significantly slower
        """
        )

    return app


if __name__ == "__main__":
    print("=" * 60)
    print("Multimodal AI Generation Studio")
    print("=" * 60)
    print()
    print("Checking available models...")
    print(f"  Image Generation (SDXL): {'✓' if HEADSHOT_AVAILABLE else '✗'}")
    print(f"  Text-to-Video (SkyReels): {'✓' if VIDEO_AVAILABLE else '✗'}")
    print(f"  Text-to-Speech (Maya1): {'✓' if VOICE_AVAILABLE else '✗'}")
    print(f"  Video Morphing (MediaPipe): {'✓' if MORPHING_AVAILABLE else '✗'}")
    print()

    # Create and launch app
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
