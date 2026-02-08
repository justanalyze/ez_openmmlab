import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ez_openmmlab.core.config_manager import ConfigManager, get_config_file
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils import toml_config

class TestConfigManager:
    def test_get_config_file_valid(self):
        """Test resolving a standard model name to a config path."""
        path = get_config_file(ModelName.RTM_DET_TINY)
        assert path.exists()
        assert "rtmdet_tiny" in str(path)

    def test_build_user_config(self, tmp_path):
        """Test building a UserConfig object."""
        manager = ConfigManager()
        dataset_toml = tmp_path / "dataset.toml"
        dataset_toml.touch()
        
        with patch("ez_openmmlab.core.config_manager.DatasetConfig.from_toml") as mock_ds:
            mock_ds.return_value = MagicMock(
                classes=["cat"],
                data_root="/data",
                train=MagicMock(ann_file="t.json", img_dir="t/"),
                val=MagicMock(ann_file="v.json", img_dir="v/"),
                test=None,
                metainfo=None
            )
            
            user_cfg = manager.build_user_config(
                model="rtmdet_tiny",
                dataset_config_path=dataset_toml,
                checkpoint_path="ckpt.pth"
            )
            
            assert user_cfg.model.name == "rtmdet_tiny"
            assert user_cfg.model.num_classes == 1

    def test_load_metadata_restricted_search(self, tmp_path):
        """Test that metadata is only loaded from immediate parent's user_config.toml."""
        manager = ConfigManager()
        checkpoint_dir = tmp_path / "ckpt"
        checkpoint_dir.mkdir()
        checkpoint = checkpoint_dir / "best.pth"
        
        # Scenario 1: No user_config.toml -> return empty metadata
        meta = manager.load_metadata_from_checkpoint(checkpoint)
        assert meta["num_classes"] is None
        
        # Scenario 2: user_config.toml in parent
        config_path = checkpoint_dir / "user_config.toml"
        # We need to use real toml_config.save_user_config or just write dummy string
        # Since we use toml_config.load_user_config inside, we should write valid TOML.
        config_path.write_text("""
[model]
name = "rtmdet_tiny"
num_classes = 5
[data]
root = "r"
train_ann = "a"
train_img = "b"
val_ann = "c"
val_img = "d"
[training]
epochs = 1
batch_size = 1
""")
        
        meta = manager.load_metadata_from_checkpoint(checkpoint)
        assert meta["num_classes"] == 5
        assert meta["model_name"] == "rtmdet_tiny"

    def test_prepare_config_file_module_qualified(self, tmp_path):
        """Test generating temp config with module qualified toml_config calls."""
        manager = ConfigManager()
        config_toml = tmp_path / "config.toml"
        config_toml.write_text("""
[model]
name = "rtmdet_tiny"
num_classes = 3
[data]
root = "r"
train_ann = "a"
train_img = "b"
val_ann = "c"
val_img = "d"
[training]
epochs = 1
batch_size = 1
""")
        
        temp_file = manager.prepare_config_file(config_toml)
        try:
            assert temp_file.exists()
            content = temp_file.read_text()
            assert "_base_ =" in content
        finally:
            manager.cleanup_temp_config(temp_file)
