from pathlib import Path
from typing import Any, Dict, Optional, Union

from ez_openmmlab.core.schema.models import ModelName


class InputValidator:
    """Consolidates all validation logic for model inputs and configurations."""

    @staticmethod
    def validate_initialization(
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]],
        using_custom_weights: bool,
        num_classes: Optional[int] = None,
        num_keypoints: Optional[int] = None,
    ) -> None:
        """Performs initial validation of provided constructor arguments."""
        is_toml = isinstance(model, (Path, str)) and str(model).endswith(".toml")

        # 1. Enforce explicit checkpoint for custom configs
        if is_toml and not using_custom_weights:
            raise ValueError(
                f"You initialized the model with a custom config ({model}) but did not "
                "explicitly provide a 'checkpoint_path'. For safety and precision during "
                "export or inference, you must explicitly specify the weights you wish to use."
            )

        # 2. Enforce explicit configuration for custom weights to prevent head size mismatches
        if checkpoint_path and using_custom_weights and not is_toml:
            if num_classes is None and num_keypoints is None:
                raise ValueError(
                    f"You provided custom weights ({checkpoint_path}) but no custom configuration. "
                    "To load a custom trained model, please provide its 'config.toml' as the 'model' argument. "
                    "Alternatively, specify 'num_classes' explicitly if using a standard model name."
                )

    @staticmethod
    def validate_augments(
        augments: Optional[Dict[str, Any]], library_family: str, engine_name: str
    ) -> None:
        """Strictly validates augmentation keys against the registry."""
        if not augments:
            return

        from ez_openmmlab.core.surgery.pipeline_patchers import (
            PipelineTransformPatcherRegistry,
        )

        supported = PipelineTransformPatcherRegistry.get_supported_augments(
            library_family
        )

        for key in augments:
            if key not in supported:
                raise ValueError(
                    f"Unsupported augmentation key '{key}' for {engine_name}. "
                    f"Available augmentations for this model: {supported}"
                )

    @staticmethod
    def validate_mmpose_weights(
        model: Union[ModelName, str, Path],
        using_custom_weights: bool,
        num_keypoints: Optional[int] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        """Pose-specific input validation for custom weights."""
        if using_custom_weights and not str(model).endswith(".toml"):
            if num_keypoints is None and num_classes is None:
                raise ValueError(
                    "MMPose models require 'num_keypoints' or 'num_classes' to be specified "
                    "when loading custom weights without a config.toml."
                )
