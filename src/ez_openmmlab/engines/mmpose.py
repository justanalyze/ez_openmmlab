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
from ez_openmmlab.core.formatters import PoseResultFormatter
from ez_openmmlab.utils.input import normalize_inputs


class EZMMPose(EZMMLab):
    """Base engine for MMPose models.

    Provides shared utilities for interacting with MMPoseInferencer.
    """

    def __init__(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
    ):
        super().__init__(model, checkpoint_path, log_level)
        self._inferencer: Optional[MMPoseInferencer] = None
        self._formatter = PoseResultFormatter()

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
    ) -> List[InferenceResult]:
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
        inputs = normalize_inputs(image_path)

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

        return self._formatter.map_results(
            all_results, inputs, self._get_class_names()
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

        # 3. No generic fallback
        return {}

    @abstractmethod
    def _init_inferencer(self, device: str, **kwargs):
        """Lazy initialization of the MMPose inferencer."""
        pass
