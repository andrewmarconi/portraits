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

try:
    from portraits.generators.smart_prompts import create_smart_template, SmartPromptTemplate

    SMART_PROMPTS_AVAILABLE = True
except ImportError:
    SMART_PROMPTS_AVAILABLE = False
    print("⚠ Smart prompts not available")

try:
    from portraits.generators.smart_prompts import create_smart_template, SmartPromptTemplate

    SMART_PROMPTS_AVAILABLE = True
except ImportError:
    SMART_PROMPTS_AVAILABLE = False
    print("⚠ Smart prompts not available")


# ============================================================================
# Image Generation Tab
# ============================================================================


def generate_image_ui(prompt, num_images, lora_path, lora_scale, clip_skip, width, height):
    """Gradio wrapper for image generation."""
    if not HEADSHOT_AVAILABLE:
        return None, "Image generation not available. Check dependencies."

    print(f"🔍 generate_image_ui called with prompt length: {len(prompt)}")
    print(f"🔍 First 200 chars: {repr(prompt[:200])}")

    try:
        # Check if prompt is a JSON array or newline-separated list
        prompts = []

        # Try to parse as JSON array first
        import json

        try:
            # Clean up prompt and try to parse as JSON
            cleaned_prompt = prompt.strip()
            if cleaned_prompt.startswith("[") and cleaned_prompt.endswith("]"):
                parsed_prompts = json.loads(cleaned_prompt)
                if isinstance(parsed_prompts, list):
                    prompts = []
                    for p in parsed_prompts:
                        prompt_str = str(p).strip()
                        # Remove any surrounding quotes that might be preserved
                        if prompt_str.startswith('"') and prompt_str.endswith('"'):
                            prompt_str = prompt_str[1:-1]
                        elif prompt_str.startswith("'") and prompt_str.endswith("'"):
                            prompt_str = prompt_str[1:-1]
                        if prompt_str:
                            prompts.append(prompt_str)
                    print(f"🔍 JSON parsing successful: {len(prompts)} prompts extracted")
        except (json.JSONDecodeError, ValueError) as e:
            # Not a valid JSON array, fall back to newline separation
            print(f"🔍 JSON parsing failed: {e}, using newline separation")
            pass

        # If JSON parsing failed or didn't return prompts, use newline separation
        if not prompts:
            prompts = [p.strip() for p in prompt.split("\n") if p.strip()]
            print(f"🔍 Newline parsing: {len(prompts)} prompts extracted")

        if len(prompts) > 1:
            # Multiple prompts - generate one image per prompt
            all_image_files = []
            total_generated = 0

            print(f"🔍 Multi-prompt mode: {len(prompts)} prompts detected")
            for i, single_prompt in enumerate(prompts):
                print(
                    f"🔍 Processing prompt {i + 1}: '{single_prompt}' (length: {len(single_prompt)})"
                )
                if len(single_prompt) > 77:
                    print(f"⚠️  Warning: Prompt {i + 1} exceeds CLIP token limit (77 tokens)")
                output_dir = generate_headshot(
                    prompt=single_prompt,
                    num_images=1,  # Always 1 image per prompt in array mode
                    lora_path=lora_path if lora_path.strip() else None,
                    lora_scale=float(lora_scale),
                    width=int(width),
                    height=int(height),
                    clip_skip=int(clip_skip),
                )

                # Find generated images for this prompt
                import glob
                from pathlib import Path

                image_files = []
                for ext in ["*.png", "*.jpg", "*.jpeg"]:
                    image_files.extend(glob.glob(str(Path(output_dir) / ext)))

                # Sort by creation time (newest first)
                image_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)

                # Take the newest image for this prompt
                if image_files:
                    all_image_files.extend(image_files[:1])
                    total_generated += 1
                    print(f"🔍 Added image: {image_files[0]}")

            print(f"🔍 Returning {len(all_image_files)} images to UI from multi-prompt mode")
            return (
                all_image_files,
                f"✓ Generated {total_generated} image(s) from {len(prompts)} prompts successfully!",
            )

        else:
            # Single prompt - use the number selector
            output_dir = generate_headshot(
                prompt=prompts[0] if prompts else prompt,
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
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
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
        
        **Multi-prompt Mode:** Enter multiple prompts to generate one image per prompt.
        - **Newline format:** One prompt per line
        - **JSON array format:** `['prompt1', 'prompt2', 'prompt3']`
        """
        )

        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="A professional headshot of a person...\n\nOr enter multiple prompts:\n\n**Newline format:**\nA professional headshot of a doctor\nA professional headshot of an artist\nA professional headshot of a teacher\n\n**JSON array format:**\n['A professional headshot of a doctor', 'A professional headshot of an artist', 'A professional headshot of a teacher']",
                    lines=8,
                )

                # Info box that shows current mode
                prompt_mode_info = gr.HTML(
                    value="<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin: 10px 0;'>📝 <strong>Single Prompt Mode</strong> - Use the Number of Images slider to generate multiple variations</div>",
                    label="Prompt Mode",
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

        # Add JavaScript to handle multi-prompt detection
        prompt_input.change(
            fn=lambda x: None,
            inputs=[prompt_input],
            outputs=[prompt_mode_info],
            js="""
            function(prompt) {
                console.log('JS: Received prompt length:', prompt.length);
                console.log('JS: First 100 chars:', prompt.substring(0, 100));
                
                let prompts = [];
                
                // Try to parse as JSON array first
                try {
                    const cleaned = prompt.trim();
                    console.log('JS: Cleaned starts with [:', cleaned.startsWith('['));
                    console.log('JS: Cleaned ends with ]:', cleaned.endsWith(']'));
                    
                    if (cleaned.startsWith('[') && cleaned.endsWith(']')) {
                        const parsed = JSON.parse(cleaned);
                        console.log('JS: Parsed type:', typeof parsed);
                        console.log('JS: Is array:', Array.isArray(parsed));
                        console.log('JS: Parsed length:', parsed ? parsed.length : 0);
                        
                        if (Array.isArray(parsed)) {
                            prompts = parsed.filter(p => p && p.toString().trim());
                            console.log('JS: Filtered prompts length:', prompts.length);
                        }
                    }
                } catch (e) {
                    console.log('JS: JSON parse error:', e);
                    // Not valid JSON, fall back to newline separation
                }
                
                // If JSON parsing failed, use newline separation
                if (prompts.length === 0) {
                    prompts = prompt.split('\\n').filter(p => p.trim());
                    console.log('JS: Newline prompts length:', prompts.length);
                }
                
                const numImagesSlider = document.querySelector('input[aria-label=\"Number of Images\"]');
                let modeHtml = '';
                
                console.log('JS: Final prompts length:', prompts.length);
                
                if (prompts.length > 1) {
                    // Multi-prompt mode
                    console.log('JS: Setting multi-prompt mode');
                    if (numImagesSlider) {
                        numImagesSlider.disabled = true;
                        numImagesSlider.value = 1;
                        const event = new Event('input', { bubbles: true });
                        numImagesSlider.dispatchEvent(event);
                    }
                    modeHtml = "<div style='padding: 10px; background-color: #e8f5e8; border-radius: 5px; margin: 10px 0; border-left: 4px solid #4caf50;'>📋 <strong>Multi-Prompt Mode</strong> - Generating 1 image per prompt (" + prompts.length + " prompts detected)</div>";
                } else {
                    // Single prompt mode
                    console.log('JS: Setting single prompt mode');
                    if (numImagesSlider) {
                        numImagesSlider.disabled = false;
                    }
                    modeHtml = "<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin: 10px 0;'>📝 <strong>Single Prompt Mode</strong> - Use the Number of Images slider to generate multiple variations</div>";
                }
                
                return [modeHtml];
            }
            """,
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
# Smart Prompts Tab
# ============================================================================


def generate_smart_prompts_ui(template, max_combinations, custom_vars):
    """Gradio wrapper for smart prompt generation."""
    if not SMART_PROMPTS_AVAILABLE:
        return None, "Smart prompts not available. Check dependencies."

    try:
        # Parse custom variables if provided
        custom_variables = {}
        if custom_vars.strip():
            for line in custom_vars.strip().split("\n"):
                if "=" in line:
                    var_name, values = line.split("=", 1)
                    var_name = var_name.strip()
                    values = [v.strip() for v in values.split(",") if v.strip()]
                    if var_name and values:
                        custom_variables[var_name] = values

        # Create smart template
        smart_template = create_smart_template(
            template=template, custom_variables=custom_variables if custom_variables else None
        )

        # Generate prompts
        max_combos = int(max_combinations) if max_combinations else None
        prompts = smart_template.generate_prompts(max_combinations=max_combos)

        # Create summary info
        total_combinations = smart_template.get_total_combinations()
        variables_info = []
        for var_name, values in smart_template.variables.items():
            source = "custom" if var_name in custom_variables else "generated"
            variables_info.append(f"• {var_name}: {len(values)} values ({source})")

        summary = f"""✅ Generated {len(prompts)} prompt(s)
📊 Total possible combinations: {total_combinations}
📝 Variables detected:
{chr(10).join(variables_info)}"""

        # Format prompts as JSON array string for easy copying to image generation
        import json

        prompts_json = json.dumps(prompts, ensure_ascii=False)
        return prompts_json, summary

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_smart_prompts_tab():
    """Create the smart prompts tab interface."""
    with gr.Tab("🧠 Smart Prompts"):
        gr.Markdown(
            """
        # AI-Powered Prompt Generation
        Generate creative prompt variations using LLM intelligence to automatically fill template variables.
        
        **Features:**
        - Automatic variable detection from {template} placeholders
        - AI-generated values for common variables (colors, styles, emotions, etc.)
        - Custom variable support
        - Batch prompt generation
        """
        )

        with gr.Row():
            with gr.Column():
                template_input = gr.Textbox(
                    label="Prompt Template",
                    placeholder="A professional headshot of a {age} {ethnicity} person with {hair_color} hair, {emotion} expression, {background} background",
                    lines=3,
                    value="A professional headshot of a {age} {ethnicity} person with {hair_color} hair, {emotion} expression, {background} background",
                )

                max_combinations_input = gr.Slider(
                    minimum=1, maximum=100, value=20, step=1, label="Maximum Prompts to Generate"
                )

                with gr.Accordion("Custom Variables (optional)", open=False):
                    gr.Markdown("""
                    **Format:** `variable_name=value1,value2,value3`
                    
                    **Examples:**
                    - `hair_color=blonde,brunette,red,black`
                    - `emotion=happy,confident,thoughtful,professional`
                    - `background=office,studio,outdoor,library`
                    """)

                    custom_vars_input = gr.Textbox(
                        label="Custom Variables",
                        placeholder="hair_color=blonde,brunette,red\nemotion=happy,confident,thoughtful",
                        lines=5,
                    )

                generate_prompts_btn = gr.Button("Generate Smart Prompts", variant="primary")

                with gr.Accordion("Template Examples", open=True):
                    gr.Markdown("""
                    **Portrait Templates:**
                    - `A {style} portrait of a {age} {ethnicity} person with {hair_color} hair, {emotion} expression`
                    - `Professional headshot of a {profession} with {background} background, {lighting} lighting`
                    
                    **Art Templates:**
                    - `A {art_style} painting of a {subject} in {setting}, {mood} atmosphere`
                    - `{camera_type} photograph of {animal} in {location}, {weather} weather`
                    """)

            with gr.Column():
                prompts_output = gr.Textbox(
                    label="Generated Prompts", lines=15, interactive=True, show_copy_button=True
                )

                status_output = gr.Textbox(label="Generation Summary", lines=8, interactive=False)

        generate_prompts_btn.click(
            fn=generate_smart_prompts_ui,
            inputs=[template_input, max_combinations_input, custom_vars_input],
            outputs=[prompts_output, status_output],
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
        - 🧠 **Smart Prompts** - AI-powered prompt generation with automatic variable filling
        """
        )

        # Create tabs
        create_image_tab()
        create_video_tab()
        create_voice_tab()
        create_morphing_tab()
        if SMART_PROMPTS_AVAILABLE:
            create_smart_prompts_tab()

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
    print(f"  Smart Prompts (AI): {'✓' if SMART_PROMPTS_AVAILABLE else '✗'}")
    print()

    # Create and launch app
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
