"""Retrieval analysis utilities."""

import numpy as np


def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    """L2-normalize embeddings.

    Args:
        embeddings: Embedding matrix with shape (num_samples, embedding_dim).

    Returns:
        Normalized embedding matrix.
    """
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    return embeddings / norms


def compute_similarity_matrix(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity matrix between query and target embeddings.

    Args:
        query_embeddings: Query embeddings with shape (num_queries, dim).
        target_embeddings: Target embeddings with shape (num_targets, dim).

    Returns:
        Similarity matrix with shape (num_queries, num_targets).
    """
    normalized_queries = normalize_embeddings(query_embeddings)
    normalized_targets = normalize_embeddings(target_embeddings)

    return normalized_queries @ normalized_targets.T


def compute_top1_accuracy(
    similarity_matrix: np.ndarray,
) -> float:
    """Compute top-1 retrieval accuracy.

    Assumes the correct match for row i is column i.

    Args:
        similarity_matrix: Similarity matrix with shape
            (num_queries, num_targets).

    Returns:
        Top-1 retrieval accuracy.
    """
    predictions = similarity_matrix.argmax(axis=1)
    targets = np.arange(similarity_matrix.shape[0])

    return float((predictions == targets).mean())
