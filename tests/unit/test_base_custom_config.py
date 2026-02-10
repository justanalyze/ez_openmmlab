from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ez_openmmlab.core.engines.engine_base import EZMMLab
from ez_openmmlab.schemas.model import ModelName


class ConcreteEZMMLab(EZMMLab):
    """Concrete implementation for testing abstract EZMMLab."""

    def _init_inferencer(self, device: str, **kwargs):
        pass

    def _run_inference(self, inputs: list, out_dir: str, show: bool, **kwargs):
        return []


def test_ezmmlab_init_with_model_name_enum():
    """Test initialization with ModelName enum."""
    model_obj = ConcreteEZMMLab(model=ModelName.RTM_DET_TINY)
    assert model_obj.model == ModelName.RTM_DET_TINY.value


def test_ezmmlab_init_with_string():
    """Test initialization with string model name."""
    model_obj = ConcreteEZMMLab(model="rtmdet_tiny")
    assert model_obj.model == "rtmdet_tiny"


def test_ezmmlab_init_with_config_path_no_checkpoint(tmp_path):
    """Test initialization with a path to a config file but no checkpoint raises ValueError."""
    config_file = tmp_path / "config.toml"
    config_file.touch()

    with pytest.raises(ValueError, match="Checkpoint path is required"):
        ConcreteEZMMLab(model=config_file)


@patch("ez_openmmlab.core.config_manager.ConfigManager.load_metadata_from_toml")
@patch("ez_openmmlab.core.config_manager.get_config_file")
@patch("ez_openmmlab.core.config_manager.toml_config.load_user_config")
def test_resolve_model_config_with_toml_valid(
    mock_load_config, mock_get_config, mock_load_meta, tmp_path
):
    """Test _resolve_model_config with valid TOML creates temp file."""
    config_file = tmp_path / "config.toml"
    config_file.touch()
    checkpoint_file = tmp_path / "epoch_10.pth"
    checkpoint_file.touch()

    # Mock the UserConfig return
    mock_user_config = MagicMock()
    mock_user_config.model.name = ModelName.RTM_DET_TINY
    mock_user_config.model.num_classes = 80
    mock_user_config.model.num_keypoints = None
    mock_user_config.data.metainfo = {}
    mock_load_config.return_value = mock_user_config

    # Mock metadata loader
    mock_load_meta.return_value = {
        "num_classes": 80,
        "num_keypoints": None,
        "metainfo": {},
        "model_name": "rtmdet_tiny",
    }

    # Mock base config path
    mock_get_config.return_value = Path("/libs/mmdetection/configs/rtmdet/tiny.py")

    model_obj = ConcreteEZMMLab(model=config_file, checkpoint_path=checkpoint_file)

    # Check if temp config was set
    assert model_obj._temp_config_file is not None
    assert model_obj._temp_config_file.exists()
    assert model_obj._temp_config_file.suffix == ".py"

    # Check content
    content = model_obj._temp_config_file.read_text()
    assert '_base_ = ["/libs/mmdetection/configs/rtmdet/tiny.py"]' in content or '/libs/mmdetection/configs/rtmdet/tiny.py' in content

    # Verify model state was updated from config
    assert model_obj.model == ModelName.RTM_DET_TINY.value
    assert model_obj.num_classes == 80


def test_temp_config_cleanup(tmp_path):
    """Test that temporary config file is deleted when object is destroyed."""
    config_file = tmp_path / "config.toml"
    config_file.touch()
    checkpoint_file = tmp_path / "epoch_10.pth"
    checkpoint_file.touch()

    with patch("ez_openmmlab.core.config_manager.toml_config.load_user_config") as mock_load:
        mock_user_config = MagicMock()
        mock_user_config.model.name = ModelName.RTM_DET_TINY
        mock_load.return_value = mock_user_config

        with patch("ez_openmmlab.core.engines.engine_base.get_config_file") as mock_get:
            mock_get.return_value = Path("/dummy.py")

            model_obj = ConcreteEZMMLab(model=config_file, checkpoint_path=checkpoint_file)
            temp_path = model_obj._temp_config_file
            assert temp_path.exists()

            # Destroy object
            del model_obj

            # File should be gone
            assert not temp_path.exists()