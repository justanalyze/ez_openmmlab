import numpy as np
import pytest
from ez_openmmlab.core.results import BaseData, Boxes, Keypoints, Masks, InferenceResult

def test_base_data_basics():
    data = np.zeros((5, 6))
    orig_shape = (480, 640)
    base = BaseData(data, orig_shape)
    
    assert len(base) == 5
    # Test indexing
    indexed = base[0]
    assert isinstance(indexed, BaseData)
    assert indexed.data.shape == (6,)

def test_boxes_properties():
    # Format: [x1, y1, x2, y2, score, label]
    data = np.array([
        [10, 20, 50, 60, 0.9, 0],
        [100, 200, 150, 260, 0.4, 1]
    ])
    boxes = Boxes(data, (480, 640))
    
    assert np.allclose(boxes.xyxy, data[:, :4])
    assert np.allclose(boxes.conf, data[:, 4])
    assert np.allclose(boxes.cls, data[:, 5])
    
    # Test filtering via indexing
    filtered = boxes[boxes.conf > 0.5]
    assert len(filtered) == 1
    assert filtered.cls[0] == 0

def test_keypoints_properties():
    # Format: [N, K, 3] -> [x, y, score]
    data = np.zeros((2, 17, 3))
    data[0, 0] = [100, 100, 0.9]
    keypoints = Keypoints(data, (480, 640))
    
    assert keypoints.xy.shape == (2, 17, 2)
    assert keypoints.conf.shape == (2, 17)
    assert keypoints.xy[0, 0, 0] == 100
    assert keypoints.conf[0, 0] == 0.9

def test_masks_properties():
    # Format: [N, H, W]
    data = np.zeros((2, 480, 640), dtype=bool)
    data[0, 10:20, 10:20] = True
    masks = Masks(data, (480, 640))
    
    assert len(masks) == 2
    assert masks.data.shape == (2, 480, 640)
    # Check if we can get segments (placeholder for now)
    assert hasattr(masks, 'xy')

def test_inference_result_container():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    res = InferenceResult(
        orig_img=img,
        path="test.jpg",
        names={0: "person"},
        speed={"inference": 10.5}
    )
    
    assert res.orig_shape == (480, 640)
    assert res.boxes is None
    assert res.speed["inference"] == 10.5
