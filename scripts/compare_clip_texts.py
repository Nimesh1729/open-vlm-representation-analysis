"""Compare one image against multiple CLIP text captions."""

import torch
import torch.nn.functional as F  # noqa: N812 please shut up

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
    """Compare image-text cosine similarities."""
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

    processor = load_clip_processor(model_name)
    model = load_clip_model(model_name)
    model.to(device)

    image_path = config["data"]["sample_image"]
    candidate_texts = config["data"]["candidate_texts"]

    image_embedding = extract_image_embedding(
        image_path=image_path,
        processor=processor,
        model=model,
        device=device,
    )

    logger.info("Image path: %s", image_path)

    for text in candidate_texts:
        text_embedding = extract_text_embedding(
            text=text,
            processor=processor,
            model=model,
            device=device,
        )

        similarity = F.cosine_similarity(
            image_embedding,
            text_embedding,
        )

        logger.info(
            "Similarity %.4f | %s",
            similarity.item(),
            text,
        )


if __name__ == "__main__":
    main()
