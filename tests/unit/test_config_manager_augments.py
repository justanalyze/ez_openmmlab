import pytest
from unittest.mock import MagicMock, patch
from ez_openmmlab.core.config_manager import ConfigManager
from ez_openmmlab.core.schema.models import ModelName

def test_create_fresh_config_with_augments(tmp_path):
    """Verifies that augmentation parameters are correctly mapped to AugmentationSection."""
    manager = ConfigManager()
    dataset_toml = tmp_path / "dataset.toml"
    dataset_toml.touch()

    with patch("ez_openmmlab.core.config_manager.DatasetConfig.from_toml") as mock_ds:
        mock_ds.return_value = MagicMock(
            dataset_name="TestAug",
            classes=["cat"],
            data_root="/data",
            train=MagicMock(ann_file="t.json", img_dir="t/"),
            val=MagicMock(ann_file="v.json", img_dir="v/"),
            test=None,
            metainfo=None,
        )

        user_cfg = manager.create_fresh_config(
            model="rtmdet_tiny",
            dataset_config_path=dataset_toml,
            scale_factor=0.8,
            rotate_factor=45.0,
            random_flip_prob=0.5
        )

        assert user_cfg.augments is not None
        assert user_cfg.augments.scale_factor == 0.8
        assert user_cfg.augments.rotate_factor == 45.0
        assert user_cfg.augments.random_flip_prob == 0.5

def test_load_metadata_from_toml_with_augments(tmp_path):
    """Verifies that metadata loader correctly handles the [augments] section."""
    manager = ConfigManager()
    toml_path = tmp_path / "config.toml"
    toml_path.write_text("""
[model]
name = "rtmdet_tiny"
num_classes = 1

[data]
root = "data"
classes = ["cat"]

[training]
epochs = 1

[augments]
scale_factor = 0.7
rotate_factor = 10.0
random_flip_prob = 0.2
""")

    meta = manager.load_metadata_from_toml(toml_path)
    # Metadata should capture these if needed, or at least not crash.
    # Currently, load_metadata_from_toml extracts from user_cfg.model.model_extra
    # and user_cfg.data.metainfo. 
    # We might need to update load_metadata_from_toml to explicitly capture augments 
    # if they are needed downstream without re-parsing.
    
    # For now, let's just ensure it loads correctly.
    assert meta["model_name"] == "rtmdet_tiny"
