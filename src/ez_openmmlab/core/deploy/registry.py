from typing import Dict, Optional

class DeployConfigRegistry:
    """Registry mapping ez_openmmlab model families to MMDeploy configuration files."""

    # Internal mapping between families and their deployment configurations
    # These paths are absolute paths INSIDE the MMDeploy Docker container
    _REGISTRY = {
        "mmdet": {
            "onnx": "/mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py",
            "tensorrt": "/mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-640x640.py",
        },
        "mmpose": {
            "onnx": "/mmdeploy/configs/mmpose/pose-detection_onnxruntime_static.py",
            "tensorrt": "/mmdeploy/configs/mmpose/pose-detection_tensorrt_static-256x192.py",
        }
    }

    def get_deploy_cfg(self, family: str, format: str) -> str:
        """Resolves the MMDeploy configuration path for a given family and format.

        Args:
            family: The model family ('mmdet', 'mmpose').
            format: The target deployment format ('onnx', 'tensorrt').

        Returns:
            The path to the MMDeploy configuration file.

        Raises:
            ValueError: If the family or format is not supported.
        """
        if family not in self._REGISTRY:
            raise ValueError(f"Unsupported model family: '{family}'. Supported: {list(self._REGISTRY.keys())}")

        family_configs = self._REGISTRY[family]
        if format not in family_configs:
            raise ValueError(f"Unsupported format: '{format}' for family '{family}'. Supported: {list(family_configs.keys())}")

        return family_configs[format]
