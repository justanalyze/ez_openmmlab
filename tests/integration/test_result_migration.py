from pathlib import Path

import pytest

from ez_openmmlab import RTMDet
from ez_openmmlab.core.inference.results import Boxes, InferenceResult
from ez_openmmlab.core.schema.models import ModelName


@pytest.mark.parametrize("model", [ModelName.RTM_DET_TINY])
def test_detector_returns_vectorized_result(model):
    detector = RTMDet(model=model)
    img_path = "demos/demo.jpg"

    results = detector.predict(img_path, device="cpu")

    assert isinstance(results, list)
    result = results[0]
    assert isinstance(result, InferenceResult)
    assert result.boxes is not None
    assert isinstance(result.boxes, Boxes)
    assert result.orig_img is not None
    assert isinstance(result.names, dict)
    assert result.path == str(Path(img_path).absolute())


@pytest.mark.parametrize("model", [ModelName.RTM_POSE_S])
def test_pose_returns_vectorized_result(model):
    from ez_openmmlab import RTMPose
    from ez_openmmlab.core.inference.results import Keypoints

    pose = RTMPose(model=model)
    img_path = "demos/demo.jpg"

    results = pose.predict(img_path, device="cpu")

    assert isinstance(results, list)
    result = results[0]
    assert isinstance(result, InferenceResult)
    assert result.keypoints is not None
    assert isinstance(result.keypoints, Keypoints)
    # Pose results also often have boxes
    assert result.boxes is not None
    assert isinstance(result.boxes, Boxes)
    assert result.orig_img is not None
