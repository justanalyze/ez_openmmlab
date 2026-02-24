from typing import Any, Dict, Type

from loguru import logger

from ez_openmmlab.core.schema.config import UserConfig


class DynamicDatasetRegistry:
    """Handles runtime creation and registration of OpenMMLab dataset classes."""

    _registered_datasets: Dict[str, Type] = {}

    @classmethod
    def register_dataset(cls, config: UserConfig, family: str) -> str:
        """Dynamically creates and registers a dataset class based on config.

        Args:
            config: The user configuration containing metainfo.
            family: The library family ('mmdet' or 'mmpose').

        Returns:
            The name of the registered dataset class.

        Raises:
            ValueError: If dataset_name is missing or already registered.
        """
        # 1. Extract metadata
        dataset_name = config.data.dataset_name
        if not dataset_name:
            raise ValueError(
                "Metadata 'dataset_name' is missing in your configuration. "
                "Please provide a unique 'dataset_name' in your dataset.toml or model configuration "
                "to enable dynamic registration."
            )

        class_name = dataset_name
        metainfo = config.data.metainfo or {}

        # 1.1 Ensure 'classes' are present in metainfo (required by base datasets)
        if config.data.classes:
            metainfo["classes"] = config.data.classes

        # 1.2 Deep convert numeric string keys to integers
        # This is critical when loading from TOML, as TOML keys are always strings.
        # MMPose utilities expect integer keys for keypoint_info and skeleton_info.
        metainfo = cls._deep_convert_numeric_keys(metainfo)

        # 2. Collision Check
        if cls._is_already_registered(class_name, family):
            raise ValueError(
                f"Dataset class '{class_name}' is already registered in {family}. "
                "Please use a unique 'dataset_name' in your dataset.toml to avoid collisions."
            )

        # 3. Create Class and Register
        if family == "mmpose":
            cls._register_mmpose_dataset(class_name, metainfo)
        elif family == "mmdet":
            cls._register_mmdet_dataset(class_name, metainfo)
        else:
            raise ValueError(f"Unsupported library family: {family}")

        logger.info(f"Dynamically registered '{class_name}' for {family} session.")
        return class_name

    @classmethod
    def _deep_convert_numeric_keys(cls, data: Any) -> Any:
        """Recursively converts string keys that look like integers into actual integers.

        Required for compatibility with OpenMMLab's internal metainfo parsing.
        """
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                # Convert numeric string keys to int
                new_key = int(k) if isinstance(k, str) and k.isdigit() else k
                new_dict[new_key] = cls._deep_convert_numeric_keys(v)
            return new_dict
        elif isinstance(data, list):
            return [cls._deep_convert_numeric_keys(item) for item in data]
        return data

    @classmethod
    def _is_already_registered(cls, name: str, family: str) -> bool:
        """Checks if a name exists in the OpenMMLab registry."""
        try:
            if family == "mmpose":
                from mmpose.registry import DATASETS
            else:
                from mmdet.registry import DATASETS
            return name in DATASETS.module_dict
        except (ImportError, AttributeError):
            return False

    @classmethod
    def _register_mmpose_dataset(cls, name: str, metainfo: Dict[str, Any]):
        """Creates and registers an MMPose dataset class.

        Validates required pose metadata: keypoint_info, skeleton_info, joint_weights, sigmas.
        """
        required_fields = ["keypoint_info", "skeleton_info", "joint_weights", "sigmas"]
        missing = [f for f in required_fields if f not in metainfo]
        if missing:
            raise ValueError(
                f"Missing required pose metadata for '{name}': {', '.join(missing)}. "
                "Custom pose datasets require complete keypoint and skeleton definitions."
            )

        from mmpose.datasets.datasets.base import BaseCocoStyleDataset
        from mmpose.registry import DATASETS

        # Create the class in memory
        DynamicClass = type(name, (BaseCocoStyleDataset,), {"METAINFO": metainfo})

        # Register it
        DATASETS.register_module(name=name, module=DynamicClass)
        cls._registered_datasets[name] = DynamicClass

    @classmethod
    def _register_mmdet_dataset(cls, name: str, metainfo: Dict[str, Any]):
        """Creates and registers an MMDet dataset class.

        Supports 'palette' for visualization colors.
        """
        if "palette" not in metainfo:
            logger.info(
                f"No 'palette' provided for '{name}'. Using default detection colors. "
                "You can specify a list of [R, G, B] colors in 'metainfo.palette' in your dataset.toml."
            )

        from mmdet.datasets import CocoDataset
        from mmdet.registry import DATASETS

        # Create the class in memory
        DynamicClass = type(name, (CocoDataset,), {"METAINFO": metainfo})

        # Register it
        DATASETS.register_module(name=name, module=DynamicClass)
        cls._registered_datasets[name] = DynamicClass