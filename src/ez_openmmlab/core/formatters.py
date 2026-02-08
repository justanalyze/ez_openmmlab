from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any
import numpy as np
from pathlib import Path

from ez_openmmlab.core.results import InferenceResult, Boxes, Keypoints

class ResultFormatter(ABC):
    """Abstract base class for formatting inference results."""

    @abstractmethod
    def map_results(
        self, results: Union[Dict[str, Any], List[Any]], inputs: Union[str, List[str]], names: Dict[int, str]
    ) -> List[InferenceResult]:
        """Maps raw inference results to vectorized InferenceResult objects."""
        pass

class DetectionResultFormatter(ResultFormatter):
    """Formatter for object detection results."""

    def map_results(
        self, results: Dict[str, Any], inputs: Union[str, List[str]], names: Dict[int, str]
    ) -> List[InferenceResult]:
        predictions = results.get("predictions", [])
        input_list = inputs if isinstance(inputs, list) else [inputs]

        # Handle mismatch case
        if not predictions and input_list:
            predictions = [
                {"labels": [], "scores": [], "bboxes": []} for _ in range(len(input_list))
            ]

        return [
            self._process_single_prediction(pred, path, names)
            for pred, path in zip(predictions, input_list)
        ]

    def _process_single_prediction(
        self, raw_pred: Dict[str, Any], img_path: str, names: Dict[int, str]
    ) -> InferenceResult:
        bboxes = np.array(raw_pred.get("bboxes", []), dtype=np.float32)
        scores = np.array(raw_pred.get("scores", []), dtype=np.float32)
        labels = np.array(raw_pred.get("labels", []), dtype=np.int32)

        if len(bboxes) > 0:
            data = np.concatenate([bboxes, scores[:, None], labels[:, None]], axis=1)
        else:
            data = np.zeros((0, 6), dtype=np.float32)

        boxes = Boxes(data, (0, 0))  # Placeholder shape for lazy loading

        return InferenceResult(
            path=str(Path(img_path).absolute()),
            names=names,
            boxes=boxes,
        )

class PoseResultFormatter(ResultFormatter):
    """Formatter for pose estimation results."""

    def map_results(
        self, results: List[Any], inputs: Union[str, List[str]], names: Dict[int, str]
    ) -> List[InferenceResult]:
        # MMPose batch format: results is a list of batch results
        # Each batch result is a dict with 'predictions' key (list of lists of dicts)
        all_flattened_preds = []
        for batch_res in results:
            if "predictions" in batch_res:
                all_flattened_preds.extend(batch_res["predictions"])

        input_list = inputs if isinstance(inputs, list) else [inputs]

        # Handle edge cases where predictions might be empty
        if not all_flattened_preds and input_list:
            all_flattened_preds = [[] for _ in range(len(input_list))]

        return [
            self._process_single_prediction(pred, path, names)
            for pred, path in zip(all_flattened_preds, input_list)
        ]

    def _process_single_prediction(
        self, raw_preds: List[Dict[str, Any]], img_path: str, names: Dict[int, str]
    ) -> InferenceResult:
        kpts_list = []
        kpt_scores_list = []
        bboxes_list = []
        bbox_scores_list = []
        labels_list = []

        for p in raw_preds:
            kpts_list.append(p.get("keypoints", []))
            kpt_scores_list.append(p.get("keypoint_scores", []))

            # Handle bbox nesting ([x1, y1, x2, y2],) common in some MMPose outputs
            raw_bbox = p.get("bbox", [])
            if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) > 0:
                inner = raw_bbox[0]
                if isinstance(inner, (list, tuple)) and len(inner) >= 4:
                    bboxes_list.append(inner[:4])
                else:
                    bboxes_list.append(raw_bbox[:4])
            else:
                bboxes_list.append([0, 0, 0, 0])

            bbox_scores_list.append(p.get("bbox_score", 0.0))
            # For person pose, label is usually 0
            labels_list.append(0)

        # Convert to NumPy
        kpts = np.array(kpts_list, dtype=np.float32)
        kpt_scores = np.array(kpt_scores_list, dtype=np.float32)
        bboxes = np.array(bboxes_list, dtype=np.float32)
        bbox_scores = np.array(bbox_scores_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int32)

        # Package Boxes: [N, 6] -> [x1, y1, x2, y2, score, label]
        if len(bboxes) > 0:
            boxes_data = np.concatenate(
                [bboxes, bbox_scores[:, None], labels[:, None]], axis=1
            )
        else:
            boxes_data = np.zeros((0, 6), dtype=np.float32)

        boxes = Boxes(boxes_data, (0, 0))

        # Package Keypoints: [N, K, 3] -> [x, y, score]
        if len(kpts) > 0:
            kpts_data = np.concatenate([kpts, kpt_scores[:, :, None]], axis=2)
        else:
            kpts_data = np.zeros((0, 0, 3), dtype=np.float32)

        keypoints = Keypoints(kpts_data, (0, 0))

        return InferenceResult(
            path=str(Path(img_path).absolute()),
            names=names,
            boxes=boxes,
            keypoints=keypoints,
        )
