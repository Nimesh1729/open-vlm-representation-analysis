"""Model loading utilities."""

from typing import Any  # fuck you again

from transformers import AutoModel, AutoProcessor  # type: ignore[import-untyped]


def load_clip_model(model_name: str) -> Any:
    """Load a pretrained CLIP-compatible model.

    Args:
        model_name: Hugging Face model checkpoint name.

    Returns:
        Loaded model in evaluation mode.
    """
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    return model


def load_clip_processor(model_name: str) -> Any:
    """Load a pretrained CLIP-compatible processor.

    Args:
        model_name: Hugging Face model checkpoint name.

    Returns:
        Loaded processor.
    """
    return AutoProcessor.from_pretrained(model_name)
