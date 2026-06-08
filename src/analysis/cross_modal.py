"""Cross-modal alignment analysis utilities."""

import numpy as np

from src.analysis.retrieval import compute_similarity_matrix


def compute_matching_similarity(
    similarity_matrix: np.ndarray,
) -> np.ndarray:
    """Extract matching image-text similarities.

    Assumes the matching text for image i is text i.

    Args:
        similarity_matrix: Image-text similarity matrix with shape
            (num_images, num_texts).

    Returns:
        Matching similarities with shape (num_samples,).
    """
    return np.diag(similarity_matrix)


def compute_non_matching_similarity(
    similarity_matrix: np.ndarray,
) -> np.ndarray:
    """Extract non-matching image-text similarities.

    Args:
        similarity_matrix: Image-text similarity matrix with shape
            (num_images, num_texts).

    Returns:
        Non-matching similarities as a flat array.
    """
    mask = ~np.eye(
        similarity_matrix.shape[0],
        dtype=bool,
    )

    return similarity_matrix[mask]


def compute_alignment_summary(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
) -> dict[str, float]:
    """Compute matching vs non-matching alignment summary.

    Args:
        image_embeddings: Image embeddings with shape (num_images, dim).
        text_embeddings: Text embeddings with shape (num_texts, dim).

    Returns:
        Alignment summary metrics.
    """
    similarity_matrix = compute_similarity_matrix(
        query_embeddings=image_embeddings,
        target_embeddings=text_embeddings,
    )

    matching = compute_matching_similarity(
        similarity_matrix,
    )

    non_matching = compute_non_matching_similarity(
        similarity_matrix,
    )

    return {
        "matching_mean": float(matching.mean()),
        "non_matching_mean": float(non_matching.mean()),
        "alignment_gap": float(matching.mean() - non_matching.mean()),
    }
