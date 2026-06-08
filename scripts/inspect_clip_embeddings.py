"""Inspect CLIP text and image embeddings."""

import torch
import torch.nn.functional as F  # noqa: N812 NO I Do what I want

from src.extraction.clip_embeddings import (
    extract_image_embedding,
    extract_text_embedding,
)
from src.models.load_model import load_clip_model, load_clip_processor
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed


def main() -> None:
    """Load CLIP and inspect image/text embedding shapes."""
    logger = get_logger(__name__)
    args = parse_args()

    config = load_config(args.config)

    model_name = config["model"]["name"]
    seed = config["system"]["seed"]
    configured_device = config["system"]["device"]

    set_seed(seed)

    device = (
        configured_device
        if configured_device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    logger.info("Model: %s", model_name)
    logger.info("Device: %s", device)

    processor = load_clip_processor(model_name)
    model = load_clip_model(model_name)
    model.to(device)

    image_path = config["data"]["sample_image"]
    text = config["data"]["sample_text"]

    text_embedding = extract_text_embedding(
        text=text,
        processor=processor,
        model=model,
        device=device,
    )

    image_embedding = extract_image_embedding(
        image_path=image_path,
        processor=processor,
        model=model,
        device=device,
    )

    similarity = F.cosine_similarity(
        text_embedding,
        image_embedding,
    )

    logger.info("Text: %s", text)
    logger.info("Image path: %s", image_path)
    logger.info("Text embedding shape: %s", tuple(text_embedding.shape))
    logger.info("Image embedding shape: %s", tuple(image_embedding.shape))
    logger.info("Cosine similarity: %.4f", similarity.item())


if __name__ == "__main__":
    main()
