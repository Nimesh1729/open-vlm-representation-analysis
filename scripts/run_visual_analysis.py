"""Run PCA and UMAP visualization for CLIP embeddings."""

from pathlib import Path

import numpy as np

from src.analysis.pca import compute_pca
from src.analysis.umap import compute_umap
from src.utils.cli import parse_args
from src.utils.logger import get_logger
from src.visualization.scatter import plot_2d_embeddings


def main() -> None:
    """Run visual analysis for image and text embeddings."""
    logger = get_logger(__name__)
    parse_args()

    embeddings_dir = Path("outputs/clip/embeddings")
    output_dir = Path("outputs/clip/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(
        embeddings_dir / "labels.npy",
        allow_pickle=True,
    )

    embedding_files = {
        "image": embeddings_dir / "image_embeddings.npy",
        "text": embeddings_dir / "text_embeddings.npy",
    }

    for embedding_name, embedding_path in embedding_files.items():
        embeddings = np.load(embedding_path)

        pca_embeddings, explained_variance = compute_pca(
            embeddings,
        )

        plot_2d_embeddings(
            embeddings_2d=pca_embeddings,
            labels=labels,
            title=f"CLIP {embedding_name.capitalize()} Embeddings PCA",
            output_path=output_dir / f"{embedding_name}_pca.png",
        )

        logger.info(
            "%s PCA explained variance ratio: %s",
            embedding_name,
            explained_variance,
        )

        umap_embeddings = compute_umap(
            embeddings,
        )

        plot_2d_embeddings(
            embeddings_2d=umap_embeddings,
            labels=labels,
            title=f"CLIP {embedding_name.capitalize()} Embeddings UMAP",
            output_path=output_dir / f"{embedding_name}_umap.png",
        )

        logger.info(
            "Saved %s visualizations.",
            embedding_name,
        )


if __name__ == "__main__":
    main()
