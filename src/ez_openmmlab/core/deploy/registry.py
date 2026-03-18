class DeployConfigRegistry:
    """Registry mapping ez_openmmlab model families to MMDeploy configuration files."""

    # Internal mapping between families and their deployment configurations
    # These paths are absolute paths INSIDE the MMDeploy Docker container
    _REGISTRY = {
        "mmdet": {
            "detection": {
                "onnx": "/root/workspace/mmdeploy/configs/mmdet/detection/detection_onnxruntime_static.py",
                "tensorrt": "/root/workspace/mmdeploy/configs/mmdet/detection/detection_tensorrt_static-640x640.py",
            },
            "instance-seg": {
                "onnx": "/root/workspace/mmdeploy/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_onnxruntime_static-640x640.py",
                "tensorrt": "/root/workspace/mmdeploy/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_tensorrt_static-640x640.py",
            },
        },
        "mmpose": {
            "rtmpose": {
                "onnx": "/root/workspace/mmdeploy/configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py",
                "tensorrt": "/root/workspace/mmdeploy/configs/mmpose/pose-detection_simcc_tensorrt_dynamic-256x192.py",
            },
            "rtmo": {
                "onnx": "/root/workspace/mmdeploy/configs/mmpose/pose-detection_rtmo_onnxruntime_dynamic.py",
                "tensorrt": "/root/workspace/mmdeploy/configs/mmpose/pose-detection_rtmo_tensorrt-fp16_dynamic-640x640.py",
            },
        },
    }

    def get_deploy_cfg(self, family: str, format: str, model_name: str = "") -> str:
        """Resolves the MMDeploy configuration path for a given family and format.

        Args:
            family: The model family ('mmdet', 'mmpose').
            format: The target deployment format ('onnx', 'tensorrt').
            model_name: The specific model name (e.g. 'rtmpose_s').

        Returns:
            The path to the MMDeploy configuration file.

        Raises:
            ValueError: If the family or format is not supported.
        """
        if family not in self._REGISTRY:
            raise ValueError(
                f"Unsupported model family: '{family}'. Supported: {list(self._REGISTRY.keys())}"
            )

        family_configs = self._REGISTRY[family]

        # Handle mmdet specialization (Detection vs Instance Segmentation)
        if family == "mmdet":
            task = "instance-seg" if "rtmdet-ins" in model_name else "detection"
            family_configs = family_configs[task]

        # Handle mmpose specialization (RTMPose vs RTMO)
        if family == "mmpose":
            task = "rtmo" if "rtmo" in model_name else "rtmpose"
            family_configs = family_configs[task]

        if format not in family_configs:
            raise ValueError(
                f"Unsupported format: '{format}' for family '{family}'. Supported: {list(family_configs.keys())}"
            )

        return family_configs[format]
