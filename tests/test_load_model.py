"""Tests for model loading utilities."""

from unittest.mock import MagicMock, patch

from src.models.load_model import load_clip_model, load_clip_processor


@patch("src.models.load_model.AutoModel.from_pretrained")
def test_load_clip_model(mock_from_pretrained: MagicMock) -> None:
    """Test CLIP model loading."""
    mock_model = MagicMock()
    mock_from_pretrained.return_value = mock_model

    model = load_clip_model("openai/clip-vit-base-patch32")

    mock_from_pretrained.assert_called_once_with("openai/clip-vit-base-patch32")
    mock_model.eval.assert_called_once()
    assert model == mock_model


@patch("src.models.load_model.AutoProcessor.from_pretrained")
def test_load_clip_processor(mock_from_pretrained: MagicMock) -> None:
    """Test CLIP processor loading."""
    mock_processor = MagicMock()
    mock_from_pretrained.return_value = mock_processor

    processor = load_clip_processor("openai/clip-vit-base-patch32")

    mock_from_pretrained.assert_called_once_with("openai/clip-vit-base-patch32")
    assert processor == mock_processor
