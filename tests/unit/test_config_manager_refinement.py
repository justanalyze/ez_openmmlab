from pathlib import Path

import pytest

from ez_openmmlab.core.config_manager import ConfigManager


class TestConfigManagerRefinement:
    def test_load_metadata_from_toml_success(self, tmp_path):
        """Test extracting metadata from a valid config.toml."""
        manager = ConfigManager()
        config_path = tmp_path / "user_config.toml"
        config_path.write_text("""
[model]
name = "rtmdet_tiny"
num_classes = 5
num_keypoints = 17
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

        metadata = manager.load_metadata_from_toml(config_path)
        assert metadata["model_name"] == "rtmdet_tiny"
        assert metadata["num_classes"] == 5
        assert metadata["num_keypoints"] == 17
        assert (
            "classes" not in metadata
        )  # Classes are in DataSection, not ModelSection in the schema

    def test_load_metadata_from_toml_with_classes(self, tmp_path):
        """Test extracting metadata including classes if present."""
        manager = ConfigManager()
        config_path = tmp_path / "user_config.toml"
        config_path.write_text("""
[model]
name = "rtmdet_tiny"
num_classes = 2
[data]
root = "r"
classes = ["cat", "dog"]
train_ann = "a"
train_img = "b"
val_ann = "c"
val_img = "d"
[training]
epochs = 1
batch_size = 1
""")

        metadata = manager.load_metadata_from_toml(config_path)
        assert metadata["num_classes"] == 2
        assert metadata["metainfo"]["classes"] == ["cat", "dog"]

    def test_load_metadata_from_toml_missing_file(self):
        """Test that non-existent file raises FileNotFoundError."""
        manager = ConfigManager()
        with pytest.raises(FileNotFoundError):
            manager.load_metadata_from_toml(Path("non_existent.toml"))
