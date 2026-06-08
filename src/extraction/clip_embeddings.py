"""CLIP embedding extraction utilities."""

from pathlib import Path
from typing import Any

import torch
from PIL import Image


def extract_text_embedding(
    text: str,
    processor: Any,
    model: Any,
    device: str,
) -> torch.Tensor:
    """Extract a CLIP text embedding."""
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        embedding = model.get_text_features(**inputs)

    return embedding


def extract_image_embedding(
    image_path: str | Path,
    processor: Any,
    model: Any,
    device: str,
) -> torch.Tensor:
    """Extract a CLIP image embedding."""
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)

    return embedding
