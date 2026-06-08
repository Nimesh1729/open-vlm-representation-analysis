"""Run joint image-text CLIP embedding visualization."""

from pathlib import Path

import numpy as np

from src.analysis.pca import compute_pca
from src.utils.logger import get_logger
from src.visualization.joint_scatter import plot_joint_image_text_embeddings


def main() -> None:
    """Visualize image and text embeddings in the same PCA space."""
    logger = get_logger(__name__)

    embeddings_dir = Path("outputs/clip/embeddings")
    output_dir = Path("outputs/clip/visualizations/joint")
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

    joint_embeddings = np.concatenate(
        [
            image_embeddings,
            text_embeddings,
        ],
        axis=0,
    )

    projected_embeddings, explained_variance = compute_pca(
        joint_embeddings,
    )

    output_path = output_dir / "joint_image_text_pca.png"

    plot_joint_image_text_embeddings(
        embeddings_2d=projected_embeddings,
        labels=labels,
        title="Joint CLIP Image-Text Embeddings PCA",
        output_path=output_path,
    )

    logger.info(
        "Joint PCA explained variance ratio: %s",
        explained_variance,
    )
    logger.info("Saved joint visualization to %s", output_path)


if __name__ == "__main__":
    main()
