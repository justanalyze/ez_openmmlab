from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

from loguru import logger


class OverrideStrategy(ABC):
    """Base class for deployment configuration override strategies."""

    @abstractmethod
    def should_apply(self, base_cfg_path: str) -> bool:
        """Determines if this strategy applies to the given configuration path."""
        pass

    @abstractmethod
    def get_overrides(self, w: int, h: int) -> str:
        """Returns the Python code snippet for the overrides."""
        pass


class SimCCOnnxStrategy(OverrideStrategy):
    """Strategy for RTMPose models using SimCC and ONNXRuntime."""

    def should_apply(self, base_cfg_path: str) -> bool:
        return "pose-detection_simcc_onnxruntime_dynamic.py" in base_cfg_path

    def get_overrides(self, w: int, h: int) -> str:
        return (
            "onnx_config = dict(\n"
            f"    input_shape=[{w}, {h}],\n"
            "    output_names=['simcc_x', 'simcc_y'],\n"
            "    dynamic_axes={\n"
            "        'input': {0: 'batch'},\n"
            "        'simcc_x': {0: 'batch'},\n"
            "        'simcc_y': {0: 'batch'}\n"
            "    })\n\n"
            "codebase_config = dict(export_postprocess=False)\n\n"
        )


class SimCCTensorRTStrategy(OverrideStrategy):
    """Strategy for RTMPose models using SimCC and TensorRT."""

    def should_apply(self, base_cfg_path: str) -> bool:
        return "pose-detection_simcc_tensorrt_dynamic-256x192.py" in base_cfg_path

    def get_overrides(self, w: int, h: int) -> str:
        return (
            "onnx_config = dict(\n"
            f"    input_shape=[{w}, {h}],\n"
            "    output_names=['simcc_x', 'simcc_y'],\n"
            "    dynamic_axes={\n"
            "        'input': {0: 'batch'},\n"
            "        'simcc_x': {0: 'batch'},\n"
            "        'simcc_y': {0: 'batch'}\n"
            "    })\n\n"
            "backend_config = dict(\n"
            "    common_config=dict(max_workspace_size=1 << 30),\n"
            "    model_inputs=[\n"
            "        dict(\n"
            "            input_shapes=dict(\n"
            "                input=dict(\n"
            f"                    min_shape=[1, 3, {h}, {w}],\n"
            f"                    opt_shape=[2, 3, {h}, {w}],\n"
            f"                    max_shape=[4, 3, {h}, {w}])))\n"
            "    ])\n\n"
            "codebase_config = dict(export_postprocess=False)\n\n"
        )


class RTMOTensorRTStrategy(OverrideStrategy):
    """Strategy for RTMO models using TensorRT."""

    def should_apply(self, base_cfg_path: str) -> bool:
        return "pose-detection_rtmo_tensorrt-fp16_dynamic-640x640.py" in base_cfg_path

    def get_overrides(self, w: int, h: int) -> str:
        return (
            "onnx_config = dict(\n"
            "    output_names=['dets', 'keypoints'],\n"
            "    dynamic_axes={\n"
            "        'input': {0: 'batch'},\n"
            "        'dets': {0: 'batch'},\n"
            "        'keypoints': {0: 'batch'}\n"
            "    })\n\n"
            "backend_config = dict(\n"
            "    common_config=dict(max_workspace_size=1 << 30),\n"
            "    model_inputs=[\n"
            "        dict(\n"
            "            input_shapes=dict(\n"
            "                input=dict(\n"
            f"                    min_shape=[1, 3, {h}, {w}],\n"
            f"                    opt_shape=[1, 3, {h}, {w}],\n"
            f"                    max_shape=[1, 3, {h}, {w}])))\n"
            "    ])\n\n"
            "codebase_config = dict(\n"
            "    post_processing=dict(\n"
            "        score_threshold=0.05,\n"
            "        iou_threshold=0.5,\n"
            "        max_output_boxes_per_class=200,\n"
            "        pre_top_k=2000,\n"
            "        keep_top_k=50,\n"
            "        background_label_id=-1,\n"
            "    ))\n\n"
        )


class TensorRTStaticStrategy(OverrideStrategy):
    """Strategy for Detection/Segmentation models using static TensorRT shapes."""

    def should_apply(self, base_cfg_path: str) -> bool:
        return any(
            pattern in base_cfg_path
            for pattern in [
                "instance-seg_rtmdet-ins_tensorrt_static-640x640.py",
                "detection_tensorrt_static-640x640.py",
            ]
        )

    def get_overrides(self, w: int, h: int) -> str:
        return (
            "backend_config = dict(\n"
            "    model_inputs=[\n"
            "        dict(\n"
            "            input_shapes=dict(\n"
            "                input=dict(\n"
            f"                    min_shape=[1, 3, {h}, {w}],\n"
            f"                    opt_shape=[1, 3, {h}, {w}],\n"
            f"                    max_shape=[1, 3, {h}, {w}])))\n"
            "    ])\n"
        )


class DeployConfigModifier:
    """Orchestrates deployment configuration overrides using registered strategies."""

    _STRATEGIES: List[OverrideStrategy] = [
        SimCCOnnxStrategy(),
        SimCCTensorRTStrategy(),
        RTMOTensorRTStrategy(),
        TensorRTStaticStrategy(),
    ]

    @classmethod
    def generate_input_resize_config(
        cls,
        base_deploy_cfg: str,
        input_size: Tuple[int, int],
        output_dir: Path,
        filename: str = "deploy_config.py",
    ) -> str:
        """Generates a customized deploy config by applying applicable override strategies.

        Args:
            base_deploy_cfg: Path to the base MMDeploy config.
            input_size: Target (width, height).
            output_dir: Host directory to save the generated config.
            filename: Target filename.

        Returns:
            The absolute path to the generated config, or base_deploy_cfg if no overrides apply.
        """
        w, h = input_size

        # Identify all matching strategies and collect their overrides
        applicable_overrides = [
            s.get_overrides(w, h)
            for s in cls._STRATEGIES
            if s.should_apply(base_deploy_cfg)
        ]

        if not applicable_overrides:
            logger.debug(f"No custom overrides applicable for: {base_deploy_cfg}")
            return base_deploy_cfg

        logger.info(
            f"Generating custom deploy config for {base_deploy_cfg} with size {input_size}"
        )

        content = f"_base_ = ['{base_deploy_cfg}']\n\n"
        content += "\n".join(applicable_overrides)

        output_path = output_dir / filename
        output_path.write_text(content)

        return str(output_path.absolute())
