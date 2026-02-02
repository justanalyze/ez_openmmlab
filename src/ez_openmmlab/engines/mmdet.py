from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import cv2

from loguru import logger
from mmdet.apis import DetInferencer
from mmdet.utils import register_all_modules

from ez_openmmlab.core.base import EZMMLab
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.core.config_loader import get_config_file
from ez_openmmlab.core.results import InferenceResult, Boxes
from ez_openmmlab.utils.download import ensure_model_checkpoint

# Force registration of MMDet modules
register_all_modules(init_default_scope=True)


class EZMMDetector(EZMMLab):
    """Abstract base class for training and inference using MMDetection."""

    def __init__(
        self,
        model_name: ModelName | str,
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
    ):
        super().__init__(model_name, checkpoint_path, log_level)
        self._inferencer: Optional[DetInferencer] = None

    def predict(
        self,
        image_path: Union[str, Path, list],
        *,
        confidence: float = 0.3,
        device: str = "cuda",
        out_dir: Optional[str] = None,
        show: bool = False,
    ) -> List[InferenceResult]:
        """Runs object detection on one or more images.

        Args:
            image_path: Path to a single image, a list of paths, or a directory.
            confidence: Confidence threshold for filtering detections (default: 0.3).
            device: Computing device ('cuda', 'cpu').
            out_dir: Directory to save visualization images.
            show: Whether to pop up a window with the result.
        """
        self._init_inferencer(device)

        logger.info(f"Running inference on: {image_path} (threshold: {confidence})")

        actual_out_dir = self._resolve_out_dir(out_dir)
        inputs = self._normalize_inputs(image_path)

        assert self._inferencer is not None, "Inferencer failed to initialize."

        # DetInferencer returns NumPy-based results by default when return_datasamples=False
        results = self._inferencer(
            inputs, out_dir=actual_out_dir, show=show, pred_score_thr=confidence
        )

        return self._map_inference_results(results, inputs)

    def _init_inferencer(self, device: str):
        """Lazy initialization of the DetInferencer."""
        if self._inferencer is None:
            config_path = get_config_file(self.model_name)
            logger.info(
                f"Initializing inferencer for model: {self.model_name} (using config: {config_path})"
            )
            self._inferencer = DetInferencer(
                model=str(config_path),
                weights=str(self.checkpoint_path),
                device=device,
            )

    def _map_inference_results(
        self, results: dict, inputs: Union[str, List[str]]
    ) -> List[InferenceResult]:
        """Maps raw MMDetection results to vectorized InferenceResult objects.

        Args:
            results: Raw dictionary output from DetInferencer.
            inputs: Original input path(s) passed to predict().

        Returns:
            A list of InferenceResult objects.
        """
        names = self._get_class_names()
        predictions = results.get("predictions", [])

        # Normalize inputs to a list for consistent iteration
        input_list = inputs if isinstance(inputs, list) else [inputs]

        # Handle cases where inferencer returns fewer predictions than inputs (should not happen normally)
        if not predictions and input_list:
            predictions = [
                {"labels": [], "scores": [], "bboxes": []} for _ in range(len(input_list))
            ]

        return [
            self._process_single_prediction(pred, path, names)
            for pred, path in zip(predictions, input_list)
        ]

    def _get_class_names(self) -> dict:
        """Retrieves class names from local metainfo or inferencer.

        Returns:
            A dictionary mapping class IDs to names.
        """
        # 1. Check local metainfo (auto-loaded from config near checkpoint)
        if self.metainfo and "classes" in self.metainfo:
            return {i: name for i, name in enumerate(self.metainfo["classes"])}

        # 2. Check inferencer model metadata (contains model's original training classes)
        if self._inferencer and hasattr(self._inferencer, "model"):
            meta = getattr(self._inferencer.model, "dataset_meta", {})
            if "classes" in meta:
                return {i: name for i, name in enumerate(meta["classes"])}

        # 3. No fallback to COCO - return empty or generic mapping
        return {}

    def _process_single_prediction(
        self, raw_pred: dict, img_path: str, names: dict
    ) -> InferenceResult:
        """Converts a single raw prediction dict into an InferenceResult object."""
        # Extract and convert components to NumPy arrays
        bboxes = np.array(raw_pred.get("bboxes", []), dtype=np.float32)
        scores = np.array(raw_pred.get("scores", []), dtype=np.float32)
        labels = np.array(raw_pred.get("labels", []), dtype=np.int32)

        # Package Boxes: [N, 6] -> [x1, y1, x2, y2, score, label]
        if len(bboxes) > 0:
            data = np.concatenate([bboxes, scores[:, None], labels[:, None]], axis=1)
        else:
            data = np.zeros((0, 6), dtype=np.float32)

        # Load original image for shape metadata and plotting support
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            logger.warning(f"Could not read image for result container: {img_path}")
            orig_img = np.zeros((100, 100, 3), dtype=np.uint8)

        boxes = Boxes(data, orig_img.shape[:2])

        return InferenceResult(
            orig_img=orig_img,
            path=str(Path(img_path).absolute()),
            names=names,
            boxes=boxes,
        )

    def _configure_model_specifics(self, config):
        """Detection specific overrides are handled by subclasses."""
        pass