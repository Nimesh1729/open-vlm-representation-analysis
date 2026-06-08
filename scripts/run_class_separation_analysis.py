"""Run class separation analysis."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.class_separation import (
    compute_class_separation,
)
from src.analysis.retrieval import (
    compute_similarity_matrix,
)
from src.utils.logger import get_logger


def main() -> None:
    """Run class separation analysis."""
    logger = get_logger(__name__)

    embeddings_dir = Path(
        "outputs/clip/embeddings",
    )

    output_dir = Path(
        "outputs/clip/analysis/class_separation",
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    results = compute_class_separation(
        similarity_matrix=similarity_matrix,
        labels=labels,
    )

    results_df = pd.DataFrame(
        results,
    )

    output_path = output_dir / "class_separation.csv"

    results_df.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "\n%s",
        results_df.sort_values(
            "separation_score",
            ascending=False,
        ),
    )

    logger.info(
        "Saved results to %s",
        output_path,
    )


if __name__ == "__main__":
    main()
