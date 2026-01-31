from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import cv2

from loguru import logger
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.base import EZMMLab
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.core.results import InferenceResult, Boxes, Keypoints
from ez_openmmlab.utils.constants import COCO_CLASSES


class EZMMPose(EZMMLab):
    """Base engine for MMPose models.

    Provides shared utilities for interacting with MMPoseInferencer.
    """

    def __init__(
        self,
        model_name: ModelName | str,
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
    ):
        super().__init__(model_name, checkpoint_path, log_level)
        self._inferencer: Optional[MMPoseInferencer] = None

    def predict(
        self,
        image_path: Union[str, Path, list],
        *,
        bbox_thr: float = 0.3,
        kpt_thr: float = 0.3,
        device: str = "cuda",
        show: bool = False,
        out_dir: Optional[str] = None,
        det_cat_ids: Optional[Union[int, list[int]]] = [0],
        **kwargs,
    ) -> Union[InferenceResult, List[InferenceResult]]:
        """Runs pose estimation on one or more images.

        Args:
            image_path: Path to a single image, a list of paths, or a directory.
            bbox_thr: Bounding box score threshold.
            kpt_thr: Keypoint score threshold.
            device: Computing device ('cuda', 'cpu').
            show: Whether to display results.
            out_dir: Directory to save visualization.
            det_cat_ids: Category IDs for detection filtering.
                See `ez_openmmlab.utils.constants.COCO_CLASSES` for ID mappings.
        """
        self._init_inferencer(device=device, det_cat_ids=det_cat_ids, **kwargs)

        if self._inferencer is None:
            raise RuntimeError(
                "Inferencer not initialized. Ensure _init_inferencer is correctly implemented."
            )

        logger.info(f"Running pose estimation on: {image_path}")

        actual_out_dir = self._resolve_out_dir(out_dir)
        inputs = self._normalize_inputs(image_path)

        inferencer_kwargs = {
            "inputs": inputs,
            "show": show,
            "out_dir": actual_out_dir if actual_out_dir else None,
            "bbox_thr": bbox_thr,
            "kpt_thr": kpt_thr,
            **kwargs,
        }

        # MMPoseInferencer is a generator. We must iterate/list it to trigger
        # the internal logic (inference, visualization, and saving).
        results_gen = self._inferencer(**inferencer_kwargs)
        all_results = list(results_gen)

        return self._map_pose_results(all_results, inputs)

    def _map_pose_results(
        self, results: list, inputs: Union[str, List[str]]
    ) -> Union[InferenceResult, List[InferenceResult]]:
        """Maps raw MMPose results to vectorized InferenceResult objects."""
        # Get class names from inferencer if possible, fallback to COCO
        names = COCO_CLASSES
        if self._inferencer and hasattr(self._inferencer, 'model'):
            meta = getattr(self._inferencer.model, 'dataset_meta', {})
            if 'classes' in meta:
                names = {i: name for i, name in enumerate(meta['classes'])}

        def _to_result(raw_preds: list, img_path: str) -> InferenceResult:
            # raw_preds is a list of dicts for one image
            # Each dict: {'keypoints': [...], 'keypoint_scores': [...], 'bbox': [...], 'bbox_score': ...}
            
            kpts_list = []
            kpt_scores_list = []
            bboxes_list = []
            bbox_scores_list = []
            labels_list = []

            for p in raw_preds:
                kpts_list.append(p.get("keypoints", []))
                kpt_scores_list.append(p.get("keypoint_scores", []))
                
                # Handle bbox nesting ([x1, y1, x2, y2],)
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
                # For person pose, label is usually 0 (person)
                labels_list.append(0)

            # Convert to NumPy
            kpts = np.array(kpts_list, dtype=np.float32)
            kpt_scores = np.array(kpt_scores_list, dtype=np.float32)
            bboxes = np.array(bboxes_list, dtype=np.float32)
            bbox_scores = np.array(bbox_scores_list, dtype=np.float32)
            labels = np.array(labels_list, dtype=np.int32)

            orig_img = cv2.imread(img_path)
            if orig_img is None:
                logger.warning(f"Could not read image for result container: {img_path}")
                orig_img = np.zeros((100, 100, 3), dtype=np.uint8)

            # Package Boxes: [N, 6] -> [x1, y1, x2, y2, score, label]
            if len(bboxes) > 0:
                boxes_data = np.concatenate([bboxes, bbox_scores[:, None], labels[:, None]], axis=1)
            else:
                boxes_data = np.zeros((0, 6), dtype=np.float32)
            
            boxes = Boxes(boxes_data, orig_img.shape[:2])

            # Package Keypoints: [N, K, 3] -> [x, y, score]
            if len(kpts) > 0:
                kpts_data = np.concatenate([kpts, kpt_scores[:, :, None]], axis=2)
            else:
                kpts_data = np.zeros((0, 0, 3), dtype=np.float32)
            
            keypoints = Keypoints(kpts_data, orig_img.shape[:2])

            return InferenceResult(
                orig_img=orig_img,
                path=str(Path(img_path).absolute()),
                names={0: "person"}, # For now, pose models usually target person
                boxes=boxes,
                keypoints=keypoints
            )

        # MMPose batch format: results is a list of batch results
        # Each batch result is a dict with 'predictions' key (list of lists of dicts)
        all_flattened_preds = []
        for batch_res in results:
            if "predictions" in batch_res:
                all_flattened_preds.extend(batch_res["predictions"])

        if isinstance(inputs, list):
            return [_to_result(p, p_path) for p, p_path in zip(all_flattened_preds, inputs)]

        return _to_result(all_flattened_preds[0] if all_flattened_preds else [], inputs)

    @abstractmethod
    def _configure_model_specifics(self, config):
        """Subclasses must implement architecture-specific overrides."""
        pass

    @abstractmethod
    def _init_inferencer(self, device: str, **kwargs):
        """Lazy initialization of the MMPose inferencer."""
        pass