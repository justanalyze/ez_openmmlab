from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from ez_openmmlab.core.inference import Boxes, InferenceResult, Keypoints
from ez_openmmlab.models.mmpose import RTMPose
from ez_openmmlab.schemas.model import ModelName


@patch("mmengine.infer.infer._load_checkpoint")
@patch("pathlib.Path.exists")
@patch("ez_openmmlab.models.mmpose.rtmpose.MMPoseInferencer")
@patch("ez_openmmlab.core.engines.engine_base.ensure_model_checkpoint")
def test_rtmpose_predict_converts_results(
    mock_ensure, mock_inferencer_cls, mock_exists, mock_load
):
    """Verifies that RTMPose correctly calls MMPoseInferencer and converts results."""
    mock_ensure.return_value = Path("dummy.pth")
    mock_exists.return_value = True
    mock_load.return_value = {"state_dict": {}}

    # Mock raw MMPose result
    raw_result = {
        "predictions": [
            [
                {
                    "keypoints": [[100, 100], [200, 200]],
                    "keypoint_scores": [0.9, 0.8],
                    "bbox": [[0, 0, 50, 50]],
                    "bbox_score": 0.95,
                }
            ]
        ]
    }

    mock_inferencer_instance = MagicMock()
    mock_inferencer_instance.return_value = iter([raw_result])
    mock_inferencer_cls.return_value = mock_inferencer_instance

    # Mock cv2.imread
    with patch("cv2.imread", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
        model_obj = RTMPose(model=ModelName.RTM_POSE_TINY)
        image_path = "demos/demo.jpg"
        results = model_obj.predict(image_path, device="cpu", bbox_thr=0.4, kpt_thr=0.4)

    # Verify inferencer was initialized
    mock_inferencer_cls.assert_called_once()

    assert isinstance(results, list)
    result = results[0]
    assert isinstance(result, InferenceResult)
    assert result.keypoints is not None
    assert isinstance(result.keypoints, Keypoints)
    assert len(result.keypoints) == 1
    assert np.allclose(result.keypoints.conf[0, 0], 0.9)

    assert result.boxes is not None
    assert isinstance(result.boxes, Boxes)
    assert np.allclose(result.boxes.conf[0], 0.95)
