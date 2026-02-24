import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from ez_openmmlab.core.config_manager import ConfigManager, get_config_file
from ez_openmmlab.core.schema.models import ModelName

class TestConfigManager:
    def test_get_config_path_standard(self):
        # get_config_file is the public utility
        path = get_config_file(ModelName.RTM_DET_TINY)
        assert path.exists()
        assert "rtmdet_tiny" in str(path)

    def test_get_config_path_invalid(self):
        with pytest.raises(ValueError, match="not found in internal map"):
            get_config_file("invalid_model")

    def test_build_user_config(self, tmp_path):
        """Test building a UserConfig object."""
        manager = ConfigManager()
        dataset_toml = tmp_path / "dataset.toml"
        dataset_toml.touch()

        with patch(
            "ez_openmmlab.core.config_manager.DatasetConfig.from_toml"
        ) as mock_ds:
            mock_ds.return_value = MagicMock(
                dataset_name="TestBuild",
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
                checkpoint_path="ckpt.pth",
            )

            assert user_cfg.model.name == "rtmdet_tiny"
            assert user_cfg.model.num_classes == 1

    def test_load_metadata_from_toml(self, tmp_path):
        """Test loading metadata from a config.toml file."""
        manager = ConfigManager()
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("""
[model]
name = "rtmpose_s"
num_classes = 10
num_keypoints = 17
input_size = [256, 256]

[data]
root = "data/coco"
classes = ["person"]

[training]
epochs = 100
""")
        
        meta = manager.load_metadata_from_toml(toml_path)
        assert meta["model_name"] == "rtmpose_s"
        assert meta["num_classes"] == 10
        assert meta["num_keypoints"] == 17
        assert meta["architecture_params"]["input_size"] == [256, 256]
        assert meta["metainfo"]["classes"] == ["person"]

    def test_prepare_config_file(self, tmp_path):
        """Test generating a temporary .py config file."""
        manager = ConfigManager()
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("""
[model]
name = "rtmdet_tiny"
num_classes = 80

[data]
root = "data/coco"

[training]
epochs = 10
""")
        
        py_path = manager.prepare_config_file(toml_path)
        assert py_path.exists()
        assert py_path.suffix == ".py"
        
        content = py_path.read_text()
        assert "_base_ =" in content
        
        manager.cleanup_temp_config(py_path)
        assert not py_path.exists()