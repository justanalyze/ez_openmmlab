from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ez_openmmlab import RTMDet
from ez_openmmlab.core.inference.results import InferenceResult


@patch("ez_openmmlab.core.resolvers.resource_resolver.ensure_model_checkpoint")
def test_predict_initializes_inferencer_and_calls_it(mock_ensure):
    """Test that predict() initializes DetInferencer with correct params
    and calls it with the provided image.
    """
    model = "rtmdet_tiny"
    checkpoint_path = "checkpoints/best.pth"
    image_path = "demo.jpg"
    mock_ensure.return_value = Path(checkpoint_path)

    # Mock result
    mock_result = {
        "predictions": [
            {"labels": [0], "scores": [0.9], "bboxes": [[10, 10, 100, 100]]}
        ]
    }

    with patch("ez_openmmlab.core.engines.mmdet.DetInferencer") as mock_inferencer_cls:
        # Configure mock inferencer instance
        mock_inferencer_instance = MagicMock()
        mock_inferencer_instance.return_value = mock_result
        mock_inferencer_cls.return_value = mock_inferencer_instance

        # Mock cv2.imread
        with patch("cv2.imread", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            # Initialize with checkpoint
            detector = RTMDet(
                model=model, checkpoint_path=checkpoint_path, num_classes=80
            )

            # Predict WITHOUT checkpoint_path
            results = detector.predict(image_path=image_path, confidence=0.5)

        # Verify DetInferencer was initialized correctly
        mock_inferencer_cls.assert_called_once()
        _, kwargs = mock_inferencer_cls.call_args

        # We expect the full path here now (as part of the Config object)
        expected_config = str(
            Path.cwd()
            / "libs/mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py"
        )
        assert str(kwargs["model"].filename) == expected_config
        assert kwargs["weights"] == checkpoint_path

        # Verify inferencer was called with the image and threshold
        mock_inferencer_instance.assert_called_once_with(
            "demo.jpg", out_dir="", show=False, pred_score_thr=0.5
        )

        # Verify results is a list of InferenceResult objects
        assert isinstance(results, list)
        assert isinstance(results[0], InferenceResult)


@patch("ez_openmmlab.core.resolvers.resource_resolver.ensure_model_checkpoint")
def test_predict_with_out_dir_creates_directory(mock_ensure, tmp_path):
    """Test that predict() passes out_dir to the inferencer."""
    model = "rtmdet_tiny"
    checkpoint_path = "checkpoints/best.pth"
    image_path = "demo.jpg"
    out_dir = tmp_path / "results"
    mock_ensure.return_value = Path(checkpoint_path)

    with patch("ez_openmmlab.core.engines.mmdet.DetInferencer") as mock_inferencer_cls:
        mock_inferencer_instance = MagicMock()
        mock_inferencer_instance.return_value = {
            "predictions": [{"labels": [], "scores": [], "bboxes": []}]
        }
        mock_inferencer_cls.return_value = mock_inferencer_instance

        # Mock cv2.imread
        with patch("cv2.imread", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            detector = RTMDet(
                model=model, checkpoint_path=checkpoint_path, num_classes=80
            )
            detector.predict(image_path=image_path, out_dir=str(out_dir))

        # Verify inferencer was called with the out_dir
        mock_inferencer_instance.assert_called_once_with(
            "demo.jpg", out_dir=str(out_dir), show=False, pred_score_thr=0.3
        )


def test_custom_weights_without_config_raises_error():
    """Verify that using custom weights requires either a .toml or explicit num_classes."""
    with pytest.raises(ValueError, match="no custom configuration"):
        RTMDet(model="rtmdet_tiny", checkpoint_path="custom.pth")
