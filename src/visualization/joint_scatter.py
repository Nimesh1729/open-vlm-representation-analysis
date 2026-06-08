"""Joint image-text embedding visualization utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_joint_image_text_embeddings(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot image and text embeddings with connecting lines.

    Assumes embeddings are ordered as:

    image_0, image_1, ..., image_n,
    text_0, text_1, ..., text_n

    Args:
        embeddings_2d: Joint 2D embeddings with shape (2 * num_samples, 2).
        labels: Labels with shape (num_samples,).
        output_path: Path where the figure should be saved.
        title: Plot title.

    Raises:
        ValueError: If embeddings and labels have incompatible shapes.
    """
    num_samples = len(labels)

    if embeddings_2d.shape[0] != 2 * num_samples:
        raise ValueError("Expected 2 * num_samples embeddings for image-text pairs.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_embeddings = embeddings_2d[:num_samples]
    text_embeddings = embeddings_2d[num_samples:]

    plt.figure(figsize=(8, 6))

    for index, label in enumerate(labels):
        image_point = image_embeddings[index]
        text_point = text_embeddings[index]

        plt.plot(
            [image_point[0], text_point[0]],
            [image_point[1], text_point[1]],
            linestyle="--",
            linewidth=1,
        )

        plt.scatter(
            image_point[0],
            image_point[1],
            marker="o",
            label=f"{label}_image",
        )

        plt.scatter(
            text_point[0],
            text_point[1],
            marker="x",
            label=f"{label}_text",
        )

        midpoint = (image_point + text_point) / 2

        plt.text(
            midpoint[0],
            midpoint[1],
            str(label),
            fontsize=9,
        )

    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
