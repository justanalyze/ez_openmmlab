from pathlib import Path
from unittest.mock import patch

import pytest

from ez_openmmlab.core.engines.engine_base import EZMMLab
from ez_openmmlab.schemas.model import ModelName


class MockEngine(EZMMLab):
    def _init_inferencer(self, device: str, **kwargs):
        pass

    def _run_inference(self, inputs: list, out_dir: str, show: bool, **kwargs):
        return []


@pytest.fixture
def mock_config_manager():
    with patch("ez_openmmlab.core.engines.engine_base.ConfigManager") as mock:
        yield mock.return_value


@pytest.fixture
def mock_ensure_checkpoint():
    with patch("ez_openmmlab.core.engines.engine_base.ensure_model_checkpoint") as mock:
        yield mock


@pytest.fixture
def mock_get_config_file():
    with patch("ez_openmmlab.core.engines.engine_base.get_config_file") as mock:
        yield mock


def test_init_standard_model(
    mock_config_manager, mock_ensure_checkpoint, mock_get_config_file
):
    """Test initialization with a standard model name."""
    model_name = ModelName.RTM_DET_TINY
    checkpoint = "checkpoints/rtmdet_tiny.pth"
    config_path = Path("/abs/path/to/config.py")

    mock_ensure_checkpoint.return_value = Path(checkpoint)
    mock_get_config_file.return_value = config_path

    with patch("ez_openmmlab.mute_warnings") as mock_mute:
        engine = MockEngine(
            model=model_name, checkpoint_path=checkpoint, num_classes=80
        )

        mock_mute.assert_called_once()
        assert engine.model == model_name.value
        assert engine.checkpoint_path == Path(checkpoint)
        assert engine.config_path == config_path
        # Metadata should NOT be auto-loaded for standard models if not present next to it
        # but EZMMLab currently still tries if checkpoint exists.
        # The new spec says standard path resolves via ensure_model_checkpoint and get_config_file.


def test_init_custom_toml_success(
    mock_config_manager, mock_ensure_checkpoint, mock_get_config_file, tmp_path
):
    """Test initialization with a custom config.toml."""
    config_toml = tmp_path / "custom.toml"
    config_toml.touch()
    checkpoint = tmp_path / "custom.pth"
    checkpoint.touch()
    temp_py_config = tmp_path / "temp_config.py"

    mock_config_manager.prepare_config_file.return_value = temp_py_config
    mock_config_manager.load_metadata_from_toml.return_value = {
        "num_classes": 10,
        "num_keypoints": 5,
        "metainfo": None,
        "model_name": "custom_model",
    }

    engine = MockEngine(model=config_toml, checkpoint_path=checkpoint)

    assert engine.model == "custom_model"
    assert engine.checkpoint_path == checkpoint
    assert engine.config_path == temp_py_config
    assert engine.num_classes == 10
    assert engine.num_keypoints == 5
    mock_config_manager.prepare_config_file.assert_called_once_with(
        config_toml, checkpoint
    )
    mock_config_manager.load_metadata_from_toml.assert_called_once_with(config_toml)


def test_init_custom_toml_missing_checkpoint(tmp_path):
    """Test initialization with config.toml but no checkpoint path raises ValueError."""
    config_toml = tmp_path / "custom.toml"
    config_toml.touch()

    with pytest.raises(ValueError, match="Checkpoint path is required"):
        MockEngine(model=config_toml)


def test_init_logging_configuration():
    """Test that logging is configured during initialization."""
    with patch("loguru.logger.remove") as mock_remove:
        with patch("loguru.logger.add") as mock_add:
            MockEngine(model=ModelName.RTM_DET_TINY, log_level="DEBUG")
            mock_remove.assert_called()
            args, kwargs = mock_add.call_args
            assert kwargs["level"] == "DEBUG"
