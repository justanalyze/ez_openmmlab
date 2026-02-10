import pytest
import numpy as np
from unittest.mock import MagicMock
from ez_openmmlab.core.formatters import DetectionResultFormatter
from ez_openmmlab.core.results import InferenceResult, Boxes

class TestDetectionResultFormatter:
    def test_map_results_empty(self):
        formatter = DetectionResultFormatter()
        results = formatter.map_results({}, [], {})
        assert len(results) == 0

    def test_map_results_single_image(self):
        formatter = DetectionResultFormatter()
        raw_results = {
            "predictions": [
                {
                    "bboxes": [[10, 10, 50, 50]],
                    "scores": [0.9],
                    "labels": [0]
                }
            ]
        }
        inputs = ["img1.jpg"]
        names = {0: "person"}
        
        # Mock cv2.imread to avoid file IO
        with pytest.MonkeyPatch.context() as m:
            m.setattr("cv2.imread", lambda x: np.zeros((100, 100, 3), dtype=np.uint8))
            
            results = formatter.map_results(raw_results, inputs, names)
            
            assert len(results) == 1
            res = results[0]
            assert isinstance(res, InferenceResult)
            assert isinstance(res.boxes, Boxes)
            assert len(res.boxes) == 1
            assert np.allclose(res.boxes.xyxy[0], [10, 10, 50, 50])
            assert np.allclose(res.boxes.conf[0], 0.9)
            assert res.boxes.cls[0] == 0

    def test_map_results_multiple_images(self):
        formatter = DetectionResultFormatter()
        raw_results = {
            "predictions": [
                {"bboxes": [], "scores": [], "labels": []},
                {"bboxes": [[0, 0, 10, 10]], "scores": [0.5], "labels": [1]}
            ]
        }
        inputs = ["img1.jpg", "img2.jpg"]
        names = {0: "cat", 1: "dog"}
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr("cv2.imread", lambda x: np.zeros((100, 100, 3), dtype=np.uint8))
            
            results = formatter.map_results(raw_results, inputs, names)
            
            assert len(results) == 2
            assert len(results[0].boxes) == 0
            assert len(results[1].boxes) == 1
            assert results[1].boxes.cls[0] == 1

    def test_map_results_segmentation(self):
        formatter = DetectionResultFormatter()
        raw_results = {
            "predictions": [
                {
                    "bboxes": [[10, 10, 50, 50]],
                    "scores": [0.9],
                    "labels": [0],
                    "masks": [np.zeros((100, 100), dtype=bool)]
                }
            ]
        }
        inputs = ["img1.jpg"]
        names = {0: "person"}
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr("cv2.imread", lambda x: np.zeros((100, 100, 3), dtype=np.uint8))
            
            results = formatter.map_results(raw_results, inputs, names)
            
            assert len(results) == 1
            res = results[0]
            assert res.masks is not None
            assert len(res.masks) == 1
            assert res.masks.data.shape == (1, 100, 100)

class TestPoseResultFormatter:
    def test_map_results_single_image(self):
        from ez_openmmlab.core.formatters import PoseResultFormatter
        from ez_openmmlab.core.results import Keypoints
        
        formatter = PoseResultFormatter()
        raw_results = [
            {
                "predictions": [
                    [  # Image 1 instances
                        {
                            "keypoints": [[100, 100], [200, 200]],
                            "keypoint_scores": [0.9, 0.8],
                            "bbox": [[0, 0, 50, 50]],
                            "bbox_score": 0.95
                        }
                    ]
                ]
            }
        ]
        inputs = ["img1.jpg"]
        names = {0: "person"}
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr("cv2.imread", lambda x: np.zeros((100, 100, 3), dtype=np.uint8))
            
            results = formatter.map_results(raw_results, inputs, names)
            
            assert len(results) == 1
            res = results[0]
            assert isinstance(res.keypoints, Keypoints)
            assert len(res.keypoints) == 1
            assert np.allclose(res.keypoints.xy[0], [[100, 100], [200, 200]])
            assert np.allclose(res.keypoints.conf[0], [0.9, 0.8])
            
            assert isinstance(res.boxes, Boxes)
            assert np.allclose(res.boxes.conf[0], 0.95)

    def test_map_results_nested_bbox(self):
        from ez_openmmlab.core.formatters import PoseResultFormatter
        
        formatter = PoseResultFormatter()
        # Some MMPose models return nested bboxes like [[x1, y1, x2, y2]]
        raw_results = [
            {
                "predictions": [
                    [
                        {
                            "keypoints": [[0, 0]], # Dummy keypoint
                            "keypoint_scores": [0.0],
                            "bbox": [[0, 0, 50, 50]], # Nested list (depth 2)
                            "bbox_score": 0.95
                        }
                    ]
                ]
            }
        ]
        inputs = ["img1.jpg"]
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr("cv2.imread", lambda x: np.zeros((100, 100, 3), dtype=np.uint8))
            results = formatter.map_results(raw_results, inputs, {})
            
            assert np.allclose(results[0].boxes.xyxy[0], [0, 0, 50, 50])
