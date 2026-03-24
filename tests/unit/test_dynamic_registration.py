import pytest
from unittest.mock import MagicMock
from ez_openmmlab.core.datasets import DynamicDatasetRegistry
from ez_openmmlab.core.schema.config import UserConfig, ModelSection, DataSection, TrainingSection

def test_dynamic_registration_mmpose():
    """Verify that a class is created and registered in mmpose."""
    config = UserConfig(
        model=ModelSection(name="rtmpose_tiny", num_classes=2, num_keypoints=5),
        data=DataSection(
            root=".",
            dataset_name="TestPose",
            metainfo={
                "classes": ["point"],
                "keypoint_info": {},
                "skeleton_info": {},
                "joint_weights": [],
                "sigmas": [],
            },
        ),
        training=TrainingSection(),
    )
    
    class_name = DynamicDatasetRegistry.register_dataset(config, "mmpose")
    assert class_name == "TestPose"
    
    from mmpose.registry import DATASETS
    assert class_name in DATASETS.module_dict
    
    # Check that the proxy class was registered
    from ez_openmmlab.core.datasets import EzMMPoseDataset
    dynamic_cls = DATASETS.get(class_name)
    assert dynamic_cls is EzMMPoseDataset

def test_dynamic_registration_mmdet():
    """Verify that a class is created and registered in mmdet."""
    config = UserConfig(
        model=ModelSection(name="rtmdet_tiny", num_classes=2),
        data=DataSection(
            root=".",
            dataset_name="TestDet",
            metainfo={"classes": ["cat"]}
        ),
        training=TrainingSection()
    )
    
    class_name = DynamicDatasetRegistry.register_dataset(config, "mmdet")
    assert class_name == "TestDet"
    
    from mmdet.registry import DATASETS
    assert class_name in DATASETS.module_dict
    
    # Check that the proxy class was registered
    from ez_openmmlab.core.datasets import EzMMDetDataset
    dynamic_cls = DATASETS.get(class_name)
    assert dynamic_cls is EzMMDetDataset

def test_duplicate_registration_is_idempotent():
    """Verify that using the same dataset_name twice does not raise error."""
    config = UserConfig(
        model=ModelSection(name="rtmdet_tiny", num_classes=2),
        data=DataSection(
            root=".",
            dataset_name="Duplicate",
            metainfo={}
        ),
        training=TrainingSection()
    )
    
    # First time should pass
    DynamicDatasetRegistry.register_dataset(config, "mmdet")
    
    # Second time should also pass (idempotent)
    class_name = DynamicDatasetRegistry.register_dataset(config, "mmdet")
    assert class_name == "Duplicate"

def test_missing_dataset_name_raises_error():
    """Verify that missing dataset_name raises ValueError."""
    config = UserConfig(
        model=ModelSection(name="rtmdet_tiny", num_classes=2),
        data=DataSection(
            root=".",
            dataset_name=None,
            metainfo={}
        ),
        training=TrainingSection()
    )
    
    with pytest.raises(ValueError, match="dataset_name' is missing"):
        DynamicDatasetRegistry.register_dataset(config, "mmdet")

def test_dataset_name_injection_into_metainfo():
    """Verify that dataset_name is automatically added to metainfo if missing."""
    config = UserConfig(
        model=ModelSection(name="rtmdet_tiny", num_classes=1),
        data=DataSection(
            root=".",
            dataset_name="InjectionTest",
            classes=["item"],
            metainfo={} # Empty metainfo
        ),
        training=TrainingSection()
    )
    
    DynamicDatasetRegistry.register_dataset(config, "mmdet")
    
    from mmdet.registry import DATASETS
    dynamic_cls = DATASETS.get("InjectionTest")
    
    # It should point to our static proxy
    from ez_openmmlab.core.datasets import EzMMDetDataset
    assert dynamic_cls is EzMMDetDataset
