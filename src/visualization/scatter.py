"""Scatter plot visualization utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_2d_embeddings(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    """Plot 2D embeddings with labels.

    Args:
        embeddings_2d: 2D embedding matrix with shape (num_samples, 2).
        labels: Label array with shape (num_samples,).
        title: Plot title.
        output_path: Path where the plot should be saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))

    unique_labels = sorted(set(labels.tolist()))

    for label in unique_labels:
        mask = labels == label

        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=label,
        )

    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
