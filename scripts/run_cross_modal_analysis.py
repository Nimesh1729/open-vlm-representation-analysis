"""Run cross-modal CLIP alignment analysis."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.cross_modal import compute_alignment_summary
from src.analysis.retrieval import compute_similarity_matrix
from src.utils.cli import parse_args
from src.utils.logger import get_logger


def main() -> None:
    """Run matching vs non-matching image-text alignment analysis."""
    logger = get_logger(__name__)
    parse_args()

    embeddings_dir = Path("outputs/clip/embeddings")
    output_dir = Path("outputs/clip/analysis/cross_modal")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_embeddings = np.load(
        embeddings_dir / "image_embeddings.npy",
    )
    text_embeddings = np.load(
        embeddings_dir / "text_embeddings.npy",
    )
    labels = np.load(
        embeddings_dir / "labels.npy",
        allow_pickle=True,
    )

    similarity_matrix = compute_similarity_matrix(
        query_embeddings=image_embeddings,
        target_embeddings=text_embeddings,
    )

    summary = compute_alignment_summary(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
    )

    summary_path = output_dir / "alignment_summary.csv"
    pairwise_path = output_dir / "pairwise_alignment.csv"

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_path, index=False)

    rows = []

    for image_index, image_label in enumerate(labels):
        for text_index, text_label in enumerate(labels):
            rows.append(
                {
                    "image_label": image_label,
                    "text_label": text_label,
                    "matching_pair": image_index == text_index,
                    "similarity": float(similarity_matrix[image_index, text_index]),
                }
            )

    pairwise_df = pd.DataFrame(rows)
    pairwise_df.to_csv(pairwise_path, index=False)

    logger.info("Alignment summary: %s", summary)
    logger.info("Saved summary to %s", summary_path)
    logger.info("Saved pairwise results to %s", pairwise_path)


if __name__ == "__main__":
    main()
