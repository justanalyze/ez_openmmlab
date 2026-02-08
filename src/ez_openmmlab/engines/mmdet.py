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
from ez_openmmlab.core.results import InferenceResult
from ez_openmmlab.utils.download import ensure_model_checkpoint
from ez_openmmlab.core.formatters import DetectionResultFormatter
from ez_openmmlab.utils.input import normalize_inputs


# Force registration of MMDet modules
register_all_modules(init_default_scope=True)


class EZMMDetector(EZMMLab):
    """Abstract base class for training and inference using MMDetection."""

    def __init__(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
    ):
        super().__init__(model, checkpoint_path, log_level)
        self._inferencer: Optional[DetInferencer] = None
        self._formatter = DetectionResultFormatter()

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
        inputs = normalize_inputs(image_path)

        assert self._inferencer is not None, "Inferencer failed to initialize."

        # DetInferencer returns NumPy-based results by default when return_datasamples=False
        results = self._inferencer(
            inputs, out_dir=actual_out_dir, show=show, pred_score_thr=confidence
        )

        return self._formatter.map_results(
            results, inputs, self._get_class_names()
        )

    def _init_inferencer(self, device: str):
        """Lazy initialization of the DetInferencer."""
        if self._inferencer is None:
            # self.config_path is set by base class
            logger.info(
                f"Initializing inferencer for model: {self.model} (using config: {self.config_path})"
            )
            self._inferencer = DetInferencer(
                model=str(self.config_path),
                weights=str(self.checkpoint_path),
                device=device,
            )

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