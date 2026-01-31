import pytest
import numpy as np
from pathlib import Path
from ez_openmmlab import RTMDet
from ez_openmmlab.core.results import InferenceResult, Boxes
from ez_openmmlab.schemas.model import ModelName

@pytest.mark.parametrize("model_name", [ModelName.RTM_DET_TINY])
def test_detector_returns_vectorized_result(model_name):
    detector = RTMDet(model_name=model_name)
    img_path = "demos/demo.jpg"
    
    result = detector.predict(img_path, device="cpu")
    
    assert isinstance(result, InferenceResult)
    assert result.boxes is not None
    assert isinstance(result.boxes, Boxes)
    assert result.orig_img is not None
    assert isinstance(result.names, dict)
    assert result.path == str(Path(img_path).absolute())

@pytest.mark.parametrize("model_name", [ModelName.RTM_POSE_S])
def test_pose_returns_vectorized_result(model_name):
    from ez_openmmlab import RTMPose
    from ez_openmmlab.core.results import Keypoints
    
    pose = RTMPose(model_name=model_name)
    img_path = "demos/demo.jpg"
    
    result = pose.predict(img_path, device="cpu")
    
    assert isinstance(result, InferenceResult)
    assert result.keypoints is not None
    assert isinstance(result.keypoints, Keypoints)
    # Pose results also often have boxes
    assert result.boxes is not None
    assert isinstance(result.boxes, Boxes)
    assert result.orig_img is not None
