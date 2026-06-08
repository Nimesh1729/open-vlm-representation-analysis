"""Run image-to-text CLIP retrieval analysis."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.retrieval import (
    compute_similarity_matrix,
    compute_top1_accuracy,
)
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


def main() -> None:
    """Run image-to-text retrieval analysis."""
    logger = get_logger(__name__)
    args = parse_args()

    config = load_config(args.config)
    captions_csv = config["data"]["captions_csv"]

    embeddings_dir = Path("outputs/clip/embeddings")
    output_dir = Path("outputs/clip/analysis/retrieval")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_embeddings = np.load(
        embeddings_dir / "image_embeddings.npy",
    )
    text_embeddings = np.load(
        embeddings_dir / "text_embeddings.npy",
    )
    captions = np.load(
        embeddings_dir / "captions.npy",
        allow_pickle=True,
    )

    dataset = pd.read_csv(captions_csv)

    similarity_matrix = compute_similarity_matrix(
        query_embeddings=image_embeddings,
        target_embeddings=text_embeddings,
    )

    top1_accuracy = compute_top1_accuracy(
        similarity_matrix,
    )

    rows = []

    predictions = similarity_matrix.argmax(axis=1)

    for query_index, predicted_index in enumerate(predictions):
        rows.append(
            {
                "query_image": dataset.iloc[query_index]["image_path"],
                "correct_caption": captions[query_index],
                "predicted_caption": captions[predicted_index],
                "correct": query_index == predicted_index,
                "similarity": float(similarity_matrix[query_index, predicted_index]),
            }
        )

    retrieval_results = pd.DataFrame(rows)

    similarity_path = output_dir / "image_to_text_similarity_matrix.npy"
    results_path = output_dir / "image_to_text_retrieval_results.csv"

    np.save(
        similarity_path,
        similarity_matrix,
    )
    retrieval_results.to_csv(
        results_path,
        index=False,
    )

    metrics_path = output_dir / "image_to_text_metrics.csv"

    metrics = pd.DataFrame(
        [
            {
                "top1_accuracy": top1_accuracy,
                "num_samples": len(retrieval_results),
                "num_correct": int(retrieval_results["correct"].sum()),
            }
        ]
    )

    metrics.to_csv(metrics_path, index=False)

    logger.info("Saved metrics to %s", metrics_path)

    logger.info("Top-1 image-to-text accuracy: %.4f", top1_accuracy)
    logger.info("Saved similarity matrix to %s", similarity_path)
    logger.info("Saved retrieval results to %s", results_path)
    logger.info("\n%s", retrieval_results)


if __name__ == "__main__":
    main()
