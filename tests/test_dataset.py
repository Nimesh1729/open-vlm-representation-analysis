"""Tests for dataset loading."""

from pathlib import Path

import pandas as pd
import pytest

from src.extraction.dataset import load_caption_dataset


def test_load_caption_dataset(
    tmp_path: Path,
) -> None:
    """Test loading a valid dataset."""
    dataset_file = tmp_path / "dataset.csv"

    pd.DataFrame(
        {
            "image_path": ["image.jpg"],
            "caption": ["A galaxy."],
            "label": ["galaxy"],
        }
    ).to_csv(
        dataset_file,
        index=False,
    )

    dataset = load_caption_dataset(
        dataset_file,
    )

    assert len(dataset) == 1
    assert dataset.iloc[0]["label"] == "galaxy"


def test_missing_columns(
    tmp_path: Path,
) -> None:
    """Test invalid dataset schema."""
    dataset_file = tmp_path / "dataset.csv"

    pd.DataFrame(
        {
            "caption": ["A galaxy."],
        }
    ).to_csv(
        dataset_file,
        index=False,
    )

    with pytest.raises(ValueError):
        load_caption_dataset(
            dataset_file,
        )
