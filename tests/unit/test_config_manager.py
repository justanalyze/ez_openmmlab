import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ez_openmmlab.core.config_manager import (
    ConfigManager,
    get_config_file,
    BaseConfigLoader,
)
from ez_openmmlab.schemas.model import ModelName, MODEL_URLS
from ez_openmmlab.utils import toml_config
from ez_openmmlab.utils.download import ensure_model_checkpoint


@pytest.fixture
def mock_project_root(tmp_path):
    """Creates a mock project root with libs/mmdetection/configs structure."""
    mmdet_root = tmp_path / "libs" / "mmdetection" / "configs"
    mmdet_root.mkdir(parents=True)
    mmpose_root = tmp_path / "libs" / "mmpose" / "configs"
    mmpose_root.mkdir(parents=True)
    return tmp_path


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

        with patch(
            "ez_openmmlab.core.config_manager.DatasetConfig.from_toml"
        ) as mock_ds:
            mock_ds.return_value = MagicMock(
                classes=["cat"],
                data_root="/data",
                train=MagicMock(ann_file="t.json", img_dir="t/"),
                val=MagicMock(ann_file="v.json", img_dir="v/"),
                test=None,
                metainfo=None,
            )

            user_cfg = manager.build_user_config(
                model="rtmdet_tiny",
                dataset_config_path=dataset_toml,
                checkpoint_path="ckpt.pth",
            )

            assert user_cfg.model.name == "rtmdet_tiny"
            assert user_cfg.model.num_classes == 1

    def test_load_metadata_from_toml(self, tmp_path):
        """Test that metadata is correctly loaded from a specific user_config.toml."""
        manager = ConfigManager()
        config_path = tmp_path / "user_config.toml"
        
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
        
        meta = manager.load_metadata_from_toml(config_path)
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

    def test_cleanup_temp_config(self, tmp_path):
        manager = ConfigManager()
        temp_file = tmp_path / "temp.py"
        temp_file.touch()
        assert temp_file.exists()

        manager.cleanup_temp_config(temp_file)
        assert not temp_file.exists()


def test_config_loader_init_validation(tmp_path):
    """Test that ConfigLoader validates the config root existence."""
    # Create instance without calling __init__ to manually set roots for testing
    loader = BaseConfigLoader.__new__(BaseConfigLoader)
    loader._mmdet_config_root = tmp_path / "non_existent"
    loader._mmpose_config_root = tmp_path / "non_existent"

    with pytest.raises(
        FileNotFoundError, match="Could not find local mmdetection or mmpose configs"
    ):
        loader._validate_root()


def test_config_loader_get_config_path_success(mock_project_root):
    """Test successful config path resolution."""
    # Create a dummy config file
    config_file = (
        mock_project_root
        / "libs"
        / "mmdetection"
        / "configs"
        / "rtmdet"
        / "rtmdet_tiny_8xb32-300e_coco.py"
    )
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.touch()

    with patch.object(BaseConfigLoader, "_validate_root", return_value=None):
        loader = BaseConfigLoader()
        loader._project_root = mock_project_root
        loader._mmdet_config_root = (
            mock_project_root / "libs" / "mmdetection" / "configs"
        )
        loader._mmpose_config_root = mock_project_root / "libs" / "mmpose" / "configs"

        path = loader.get_config_path("rtmdet_tiny")
        assert path.resolve() == config_file.resolve()


def test_config_loader_invalid_model():
    """Test ConfigLoader with an unsupported model name."""
    with patch.object(BaseConfigLoader, "_validate_root", return_value=None):
        loader = BaseConfigLoader()
        with pytest.raises(ValueError, match="not found in internal map"):
            loader.get_config_path("invalid_model")


def test_config_loader_missing_file(mock_project_root):
    """Test ConfigLoader when the mapped file is missing on disk."""
    with patch.object(BaseConfigLoader, "_validate_root", return_value=None):
        loader = BaseConfigLoader()
        loader._mmdet_config_root = (
            mock_project_root / "libs" / "mmdetection" / "configs"
        )
        loader._mmpose_config_root = mock_project_root / "libs" / "mmpose" / "configs"

        with pytest.raises(FileNotFoundError, match="not found at"):
            loader.get_config_path("rtmdet_tiny")


@patch("ez_openmmlab.utils.download.download_checkpoint")
def test_ensure_model_checkpoint_existing(mock_download, tmp_path):
    """Test that ensure_model_checkpoint returns path if file exists."""
    # Mock Path.home() to point to our tmp_path
    with patch("pathlib.Path.home", return_value=tmp_path):
        checkpoint_dir = tmp_path / ".cache" / "ez_openmmlab" / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "rtmdet_tiny.pth"
        checkpoint_file.touch()

        path = ensure_model_checkpoint("rtmdet_tiny")
        assert path == checkpoint_file
        mock_download.assert_not_called()


@patch("ez_openmmlab.utils.download.download_checkpoint")
def test_ensure_model_checkpoint_download_trigger(mock_download, tmp_path):
    """Test that ensure_model_checkpoint triggers download if file is missing."""
    with patch("pathlib.Path.home", return_value=tmp_path):
        path = ensure_model_checkpoint("rtmdet_tiny")
        expected_path = (
            tmp_path / ".cache" / "ez_openmmlab" / "checkpoints" / "rtmdet_tiny.pth"
        )
        assert path == expected_path
        mock_download.assert_called_once_with(MODEL_URLS["rtmdet_tiny"], expected_path)


def test_ensure_model_checkpoint_missing_url(tmp_path):
    """Test ensure_model_checkpoint when model has no URL and path is missing."""
    with patch("pathlib.Path.home", return_value=tmp_path):
        # Case 1: Custom path provided, file missing -> raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            ensure_model_checkpoint("unknown_model", checkpoint_path="missing.pth")

        # Case 2: No path provided, unknown model -> return path but log warning (non-fatal)
        path = ensure_model_checkpoint("unknown_model")
        assert (
            path
            == tmp_path
            / ".cache"
            / "ez_openmmlab"
            / "checkpoints"
            / "unknown_model.pth"
        )