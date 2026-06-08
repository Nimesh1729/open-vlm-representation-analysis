"""Class separation analysis utilities."""

import numpy as np


def compute_class_separation(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, float | str]]:
    """Compute within-class and between-class similarities.

    Args:
        similarity_matrix: Similarity matrix.
        labels: Labels corresponding to rows/columns.

    Returns:
        List of class separation metrics.
    """
    results = []

    unique_labels = sorted(
        set(labels.tolist()),
    )

    for target_label in unique_labels:
        within_similarities = []
        between_similarities = []

        for row_index, row_label in enumerate(labels):
            for column_index, column_label in enumerate(labels):
                if row_index == column_index:
                    continue

                similarity = similarity_matrix[
                    row_index,
                    column_index,
                ]

                if row_label == target_label and column_label == target_label:
                    within_similarities.append(
                        similarity,
                    )

                elif row_label == target_label and column_label != target_label:
                    between_similarities.append(
                        similarity,
                    )

        within_mean = float(
            np.mean(within_similarities),
        )

        between_mean = float(
            np.mean(between_similarities),
        )

        results.append(
            {
                "label": target_label,
                "within_similarity": within_mean,
                "between_similarity": between_mean,
                "separation_score": (within_mean - between_mean),
            }
        )

    return results
