"""Generation modules for Portraits."""

# Make all imports conditional to avoid dependency issues
_generate_headshot = None
_generate_video = None
_generate_voice = None
_create_mesh_morphing_video = None

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


# Set the main functions if available
generate_headshot = _generate_headshot
generate_video = _generate_video
generate_voice = _generate_voice
create_mesh_morphing_video = _create_mesh_morphing_video

__all__ = [
    "generate_headshot",
    "generate_video",
    "generate_voice",
    "create_mesh_morphing_video",
    "get_available_generators",
]
