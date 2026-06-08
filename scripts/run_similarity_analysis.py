"""Run CLIP image-text similarity matrix analysis."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.retrieval import compute_similarity_matrix
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


def main() -> None:
    """Compute and save image-text similarity matrix."""
    logger = get_logger(__name__)
    args = parse_args()

    config = load_config(args.config)
    captions_csv = config["data"]["captions_csv"]

    embeddings_dir = Path("outputs/clip/embeddings")
    output_dir = Path("outputs/clip/analysis/similarity")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_embeddings = np.load(embeddings_dir / "image_embeddings.npy")
    text_embeddings = np.load(embeddings_dir / "text_embeddings.npy")

    dataset = pd.read_csv(captions_csv)

    similarity_matrix = compute_similarity_matrix(
        query_embeddings=image_embeddings,
        target_embeddings=text_embeddings,
    )

    labels = dataset["label"].tolist()

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=[f"image_{label}" for label in labels],
        columns=[f"text_{label}" for label in labels],
    )

    matrix_path = output_dir / "image_text_similarity_matrix.npy"
    csv_path = output_dir / "image_text_similarity_matrix.csv"

    np.save(matrix_path, similarity_matrix)
    similarity_df.to_csv(csv_path)

    logger.info("Saved similarity matrix to %s", matrix_path)
    logger.info("Saved similarity CSV to %s", csv_path)
    logger.info("\n%s", similarity_df)


if __name__ == "__main__":
    main()
