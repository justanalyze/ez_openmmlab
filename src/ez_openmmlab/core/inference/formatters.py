from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from .results import Boxes, InferenceResult, Keypoints, Masks


class ResultFormatter(ABC):
    """Abstract base class for formatting inference results."""

    @abstractmethod
    def map_results(
        self,
        results: Union[Dict[str, Any], List[Any]],
        inputs: Union[str, List[str]],
        names: Dict[int, str],
    ) -> List[InferenceResult]:
        """Maps raw inference results to vectorized InferenceResult objects."""
        pass

    @abstractmethod
    def format(self, result: InferenceResult, attr: str, raw_data: Any) -> Any:
        """Formats a specific attribute from raw data."""
        pass


class DetectionResultFormatter(ResultFormatter):
    """Formatter for object detection results."""

    def map_results(
        self,
        results: Dict[str, Any],
        inputs: Union[str, List[str]],
        names: Dict[int, str],
    ) -> List[InferenceResult]:
        predictions = results.get("predictions", [])
        input_list = inputs if isinstance(inputs, list) else [inputs]

        # Handle mismatch case
        if not predictions and input_list:
            predictions = [
                {"labels": [], "scores": [], "bboxes": [], "masks": []}
                for _ in range(len(input_list))
            ]

        return [
            self._process_single_prediction(pred, path, names)
            for pred, path in zip(predictions, input_list)
        ]

    def _process_single_prediction(
        self, raw_pred: Dict[str, Any], img_path: str, names: Dict[int, str]
    ) -> InferenceResult:
        return InferenceResult(
            path=str(Path(img_path).absolute()),
            names=names,
            boxes=raw_pred,
            masks=raw_pred,
            format_fn=self.format,
        )

    def format(self, result: InferenceResult, attr: str, raw_data: Any) -> Any:
        if attr == "boxes":
            bboxes = np.array(raw_data.get("bboxes", []), dtype=np.float32)
            scores = np.array(raw_data.get("scores", []), dtype=np.float32)
            labels = np.array(raw_data.get("labels", []), dtype=np.int32)

            if len(bboxes) > 0:
                data = np.concatenate(
                    [bboxes, scores[:, None], labels[:, None]], axis=1
                )
            else:
                data = np.zeros((0, 6), dtype=np.float32)

            return Boxes(data, result.orig_shape)

        if attr == "masks":
            masks_data = raw_data.get("masks", [])
            if len(masks_data) > 0:
                # Handle RLE format (dicts) or binary masks (numpy)
                decoded_masks = []
                for m in masks_data:
                    if isinstance(m, dict) and "counts" in m:
                        import pycocotools.mask as mask_utils

                        decoded_masks.append(mask_utils.decode(m))
                    else:
                        decoded_masks.append(m)

                return Masks(np.array(decoded_masks), result.orig_shape)
            return Masks(np.zeros((0, 0, 0), dtype=bool), result.orig_shape)

        return None


class PoseResultFormatter(ResultFormatter):
    """Formatter for pose estimation results."""

    def map_results(
        self,
        results: List[Any],
        inputs: Union[str, List[str]],
        names: Dict[int, str],
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
        self,
        raw_preds: List[Dict[str, Any]],
        img_path: str,
        names: Dict[int, str],
    ) -> InferenceResult:
        return InferenceResult(
            path=str(Path(img_path).absolute()),
            names=names,
            boxes=raw_preds,
            keypoints=raw_preds,
            format_fn=self.format,
        )

    def format(self, result: InferenceResult, attr: str, raw_data: Any) -> Any:
        if attr == "boxes":
            bboxes_list = []
            bbox_scores_list = []
            labels_list = []

            for p in raw_data:
                # ... handle bounding box logic ...
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
                # Use category_id if provided (common in multi-class setups), fallback to 0
                labels_list.append(p.get("category_id", 0))

            bboxes = np.array(bboxes_list, dtype=np.float32)
            bbox_scores = np.array(bbox_scores_list, dtype=np.float32)
            labels = np.array(labels_list, dtype=np.int32)

            if len(bboxes) > 0:
                boxes_data = np.concatenate(
                    [bboxes, bbox_scores[:, None], labels[:, None]], axis=1
                )
            else:
                boxes_data = np.zeros((0, 6), dtype=np.float32)

            return Boxes(boxes_data, result.orig_shape)

        if attr == "keypoints":
            kpts_list = []
            kpt_scores_list = []

            for p in raw_data:
                kpts_list.append(p.get("keypoints", []))
                kpt_scores_list.append(p.get("keypoint_scores", []))

            kpts = np.array(kpts_list, dtype=np.float32)
            kpt_scores = np.array(kpt_scores_list, dtype=np.float32)

            if len(kpts) > 0:
                kpts_data = np.concatenate([kpts, kpt_scores[:, :, None]], axis=2)
            else:
                kpts_data = np.zeros((0, 0, 3), dtype=np.float32)

            return Keypoints(kpts_data, result.orig_shape)

        return None
