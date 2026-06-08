"""Dataset loading utilities."""

from pathlib import Path

import pandas as pd


def load_caption_dataset(
    csv_path: str | Path,
) -> pd.DataFrame:
    """Load an image-caption dataset.

    Args:
        csv_path: Path to dataset CSV.

    Returns:
        Dataset DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    dataset = pd.read_csv(csv_path)

    required_columns = {
        "image_path",
        "caption",
        "label",
    }

    if not required_columns.issubset(
        dataset.columns,
    ):
        raise ValueError("CSV must contain image_path, caption, and label columns.")

    return dataset
