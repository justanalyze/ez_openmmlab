import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ez_openmmlab.core.config_manager import ConfigManager
from ez_openmmlab.utils.toml_config import UserConfig

def test_build_user_config_defaults():
    """Verify default values for new parameters in build_user_config."""
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
        
        cfg = manager.build_user_config(
            model="rtmpose_s",
            dataset_config_path="dummy.toml"
        )
        
        # Verify defaults
        assert cfg.model.input_size == (192, 256)
        assert cfg.model.simcc_sigma == (4.9, 5.66)
        assert cfg.model.feature_map_size == (6, 8) # 192//32, 256//32
        assert cfg.training.weight_decay == 0.05
        assert cfg.training.evaluator_metric == "CocoMetric"

def test_build_user_config_auto_scaling_sigma():
    """Verify linear scaling of simcc_sigma when input_size changes."""
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
        
        # Double the resolution
        cfg = manager.build_user_config(
            model="rtmpose_s",
            dataset_config_path="dummy.toml",
            input_size=(384, 512)
        )
        
        # Sigma should also double
        assert cfg.model.input_size == (384, 512)
        assert cfg.model.simcc_sigma == (9.8, 11.32) # (4.9*2, 5.66*2)
        assert cfg.model.feature_map_size == (12, 16) # 384//32, 512//32

def test_build_user_config_explicit_overrides():
    """Verify that explicit overrides are respected."""
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
        
        cfg = manager.build_user_config(
            model="rtmpose_s",
            dataset_config_path="dummy.toml",
            input_size=(384, 512),
            simcc_sigma=(5.0, 5.0),
            feature_map_size=(10, 10),
            weight_decay=0.01,
            evaluator_metric=["PCKAccuracy", "AUC"]
        )
        
        assert cfg.model.simcc_sigma == (5.0, 5.0)
        assert cfg.model.feature_map_size == (10, 10)
        assert cfg.training.weight_decay == 0.01
        assert cfg.training.evaluator_metric == ["PCKAccuracy", "AUC"]