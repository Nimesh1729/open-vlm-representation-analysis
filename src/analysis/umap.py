"""UMAP analysis utilities."""

import numpy as np
from umap import UMAP


def compute_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Compute UMAP projection.

    Args:
        embeddings: Embedding matrix with shape (num_samples, embedding_dim).
        n_components: Number of UMAP dimensions.
        random_state: Random seed.

    Returns:
        Projected embeddings with shape (num_samples, n_components).
    """
    reducer = UMAP(
        n_components=n_components,
        random_state=random_state,
    )

    return reducer.fit_transform(embeddings)
