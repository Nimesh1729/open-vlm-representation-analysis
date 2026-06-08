"""Extract CLIP embeddings for the image-caption dataset."""

from pathlib import Path

import numpy as np
import torch

from src.extraction.dataset import load_caption_dataset
from src.extraction.dataset_embeddings import extract_dataset_embeddings
from src.models.load_model import load_clip_model, load_clip_processor
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed


def main() -> None:
    """Extract and save dataset-level CLIP embeddings."""
    logger = get_logger(__name__)
    args = parse_args()

    config = load_config(args.config)

    model_name = config["model"]["name"]
    seed = config["system"]["seed"]
    configured_device = config["system"]["device"]
    captions_csv = config["data"]["captions_csv"]

    set_seed(seed)

    device = (
        configured_device
        if configured_device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    logger.info("Model: %s", model_name)
    logger.info("Device: %s", device)
    logger.info("Dataset: %s", captions_csv)

    processor = load_clip_processor(model_name)
    model = load_clip_model(model_name)
    model.to(device)

    dataset = load_caption_dataset(captions_csv)

    embeddings = extract_dataset_embeddings(
        dataset=dataset,
        processor=processor,
        model=model,
        device=device,
    )

    output_dir = Path("outputs/clip/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        output_dir / "image_embeddings.npy",
        embeddings["image_embeddings"],
    )
    np.save(
        output_dir / "text_embeddings.npy",
        embeddings["text_embeddings"],
    )
    np.save(
        output_dir / "labels.npy",
        embeddings["labels"],
    )
    np.save(
        output_dir / "captions.npy",
        embeddings["captions"],
    )

    logger.info(
        "Image embeddings shape: %s",
        embeddings["image_embeddings"].shape,
    )
    logger.info(
        "Text embeddings shape: %s",
        embeddings["text_embeddings"].shape,
    )
    logger.info("Saved embeddings to %s", output_dir)


if __name__ == "__main__":
    main()
