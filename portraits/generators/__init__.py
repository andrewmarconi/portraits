"""Generation modules for Portraits."""

# Make all imports conditional to avoid dependency issues
_generate_headshot = None
_generate_video = None
_generate_voice = None
_create_mesh_morphing_video = None
_prompt_template = None

# Image generation
try:
    from .image import generate_headshot

    _generate_headshot = generate_headshot
except ImportError as e:
    print(f"Warning: Image generation unavailable: {e}")

# Video generation
try:
    from .video import generate_video

    _generate_video = generate_video
except ImportError as e:
    print(f"Warning: Video generation unavailable: {e}")

# Voice generation
try:
    from .voice import generate_voice

    _generate_voice = generate_voice
except ImportError as e:
    print(f"Warning: Voice generation unavailable: {e}")

# Morph generation
try:
    from .morph import create_mesh_morphing_video

    _create_mesh_morphing_video = create_mesh_morphing_video
except ImportError as e:
    print(f"Warning: Morph generation unavailable: {e}")

# Prompt template generation
try:
    from .smart_prompts import (
        SmartPromptTemplate,
        LLMVariableGenerator,
        create_smart_template,
    )

    _prompt_template = {
        "SmartPromptTemplate": SmartPromptTemplate,
        "LLMVariableGenerator": LLMVariableGenerator,
        "create_smart_template": create_smart_template,
    }
except ImportError as e:
    print(f"Warning: Prompt template generation unavailable: {e}")
    _prompt_template = {}


# Export available functions
def get_available_generators():
    """Get dictionary of available generators."""
    generators = {}
    if _generate_headshot:
        generators["image"] = _generate_headshot
    if _generate_video:
        generators["video"] = _generate_video
    if _generate_voice:
        generators["voice"] = _generate_voice
    if _create_mesh_morphing_video:
        generators["morph"] = _create_mesh_morphing_video
    return generators


def get_prompt_tools():
    """Get dictionary of available prompt tools."""
    return _prompt_template or {}


# Set the main functions if available
generate_headshot = _generate_headshot
generate_video = _generate_video
generate_voice = _generate_voice
create_mesh_morphing_video = _create_mesh_morphing_video

# Set prompt tools if available
if _prompt_template:
    SmartPromptTemplate = _prompt_template["SmartPromptTemplate"]
    LLMVariableGenerator = _prompt_template["LLMVariableGenerator"]
    create_smart_template = _prompt_template["create_smart_template"]

__all__ = [
    "generate_headshot",
    "generate_video",
    "generate_voice",
    "create_mesh_morphing_video",
    "get_available_generators",
    "get_prompt_tools",
    "SmartPromptTemplate",
    "LLMVariableGenerator",
    "create_smart_template",
]
