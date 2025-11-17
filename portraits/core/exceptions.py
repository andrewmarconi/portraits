"""Custom exceptions for Portraits."""


class PortraitsError(Exception):
    """Base exception for all Portraits errors."""

    pass


class ConfigurationError(PortraitsError):
    """Configuration-related errors."""

    pass


class ModelNotFoundError(PortraitsError):
    """Model weights not found at specified path."""

    def __init__(self, model_path: str, instructions: str = ""):
        self.model_path = model_path
        message = f"Model not found at '{model_path}'."
        if instructions:
            message += f"\n\n{instructions}"
        super().__init__(message)


class DeviceError(PortraitsError):
    """Device-related errors (CUDA, MPS, CPU)."""

    pass


class GenerationError(PortraitsError):
    """Error during generation process."""

    pass


class FaceDetectionError(PortraitsError):
    """Face detection failed in morphing."""

    pass


def handle_model_load_error(error: Exception, model_path: str) -> None:
    """Handle model loading errors with helpful messages.

    Args:
        error: Original exception
        model_path: Path where model was expected

    Raises:
        ModelNotFoundError: With helpful installation instructions
        GenerationError: For other errors
    """
    error_msg = str(error)

    if "404" in error_msg or "Entry Not Found" in error_msg:
        instructions = (
            "Please download the model manually:\n"
            f"  git lfs install\n"
            f"  git clone https://huggingface.co/{model_path}"
        )
        raise ModelNotFoundError(model_path, instructions) from error

    raise GenerationError(f"Failed to load model: {error_msg}") from error
