from unittest.mock import MagicMock, patch
from ez_openmmlab.core.config_manager import ConfigManager
import pytest

def test_config_manager_architecture_extraction():
    """Verify that architecture params are correctly extracted from UserConfig."""
    manager = ConfigManager()
    
    with patch("ez_openmmlab.core.config_manager.DatasetConfig.from_toml") as mock_ds:
        mock_ds.return_value = MagicMock(
            dataset_name="TestDS",
            classes=["cat"],
            data_root="/data",
            train=MagicMock(ann_file="t.json", img_dir="t/"),
            val=MagicMock(ann_file="v.json", img_dir="v/"),
            test=None,
            metainfo=None
        )
        
        # Test with RTMPose specific architecture params
        cfg = manager.create_fresh_config(
            model="rtmpose_s",
            dataset_config_path="dummy.toml",
            architecture_params={"input_size": (256, 256), "simcc_sigma": (5.0, 5.0)}
        )
        
        # Extract model_extra
        assert cfg.model.input_size == (256, 256)
        assert cfg.model.simcc_sigma == (5.0, 5.0)

def test_build_user_config_delegation():
    """Verify that ConfigManager correctly uses the resolver factory."""
    manager = ConfigManager()

    with patch("ez_openmmlab.core.config_manager.DatasetConfig.from_toml") as mock_ds:
        mock_ds.return_value = MagicMock(
            dataset_name="TestDS",
            classes=["cat"],
            data_root="/data",
            train=MagicMock(ann_file="t.json", img_dir="t/"),
            val=MagicMock(ann_file="v.json", img_dir="v/"),
            test=None,
            metainfo=None
        )

        # This should trigger RTMPoseParamsResolver logic
        cfg = manager.create_fresh_config(
            model="rtmpose_s",
            dataset_config_path="dummy.toml",
            architecture_params={"input_size": (170, 240)},
            weight_decay=0.01,
            evaluator_metric="PCKAccuracy"
        )
        
        assert cfg.model.input_size == (160, 256)
        assert cfg.training.weight_decay == 0.01
        assert cfg.training.evaluator_metric == "PCKAccuracy"