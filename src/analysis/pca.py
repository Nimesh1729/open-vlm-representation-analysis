"""PCA analysis utilities."""

import numpy as np
from sklearn.decomposition import PCA


def compute_pca(
    embeddings: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PCA projection.

    Args:
        embeddings: Embedding matrix with shape (num_samples, embedding_dim).
        n_components: Number of PCA components.

    Returns:
        Tuple of projected embeddings and explained variance ratio.
    """
    pca = PCA(n_components=n_components)

    projected_embeddings = pca.fit_transform(embeddings)

    return projected_embeddings, pca.explained_variance_ratio_
