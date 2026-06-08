"""Dataset-level CLIP embedding extraction utilities."""

from typing import Any

import numpy as np
import pandas as pd
import torch

from src.extraction.clip_embeddings import (
    extract_image_embedding,
    extract_text_embedding,
)


def extract_dataset_embeddings(
    dataset: pd.DataFrame,
    processor: Any,
    model: Any,
    device: str,
) -> dict[str, np.ndarray]:
    """Extract CLIP image and text embeddings for a dataset.

    Args:
        dataset: DataFrame with image_path, caption, and label columns.
        processor: CLIP processor.
        model: CLIP model.
        device: Device name.

    Returns:
        Dictionary containing image embeddings, text embeddings, labels, and captions.
    """
    image_embeddings = []
    text_embeddings = []

    captions = dataset["caption"].to_numpy()
    labels = dataset["label"].to_numpy()

    for _, row in dataset.iterrows():
        image_embedding = extract_image_embedding(
            image_path=row["image_path"],
            processor=processor,
            model=model,
            device=device,
        )

        text_embedding = extract_text_embedding(
            text=row["caption"],
            processor=processor,
            model=model,
            device=device,
        )

        image_embeddings.append(image_embedding.detach().cpu())
        text_embeddings.append(text_embedding.detach().cpu())

    image_embedding_matrix = torch.cat(
        image_embeddings,
        dim=0,
    ).numpy()

    text_embedding_matrix = torch.cat(
        text_embeddings,
        dim=0,
    ).numpy()

    return {
        "image_embeddings": image_embedding_matrix,
        "text_embeddings": text_embedding_matrix,
        "labels": labels,
        "captions": captions,
    }
