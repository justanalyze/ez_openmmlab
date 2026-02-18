import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from ez_openmmlab.core.engines.engine_base import EZMMLab
from ez_openmmlab.schemas.model import ModelName

# Concrete implementation for testing abstract base class
class MockEngine(EZMMLab):
    def _init_inferencer(self, device: str, **kwargs) -> None:
        pass
    def _run_inference(self, inputs: list, out_dir: str, show: bool, **kwargs):
        return []
    def _get_library_family(self) -> str:
        return "mmdet"
    def _get_architecture_params(self, **kwargs):
        return {}

def test_init_standard_model():
    """Test initialization with standard model name."""
    with patch("ez_openmmlab.core.engines.engine_base.ensure_model_checkpoint") as mock_ensure:
        mock_ensure.return_value = Path("dummy.pth")
        engine = MockEngine(model="rtmdet_tiny")
        assert engine.model == "rtmdet_tiny"
        assert engine.checkpoint_path == Path("dummy.pth")

def test_init_custom_toml_with_checkpoint(tmp_path):
    """Test initialization with config.toml and explicit checkpoint."""
    config_toml = tmp_path / "custom.toml"
    config_toml.write_text("""
[model]
name = "rtmdet_tiny"
num_classes = 10

[data]
root = "data/coco"

[training]
epochs = 10
""")
    checkpoint = tmp_path / "manual.pth"
    checkpoint.touch()
    
    engine = MockEngine(model=config_toml, checkpoint_path=checkpoint)
    assert engine.checkpoint_path == checkpoint
    assert engine.num_classes == 10

def test_init_custom_toml_smart_resolution(tmp_path):
    """Test initialization with config.toml and smart checkpoint resolution."""
    config_toml = tmp_path / "custom.toml"
    config_toml.write_text("""
[model]
name = "rtmdet_tiny"
num_classes = 10

[data]
root = "data/coco"

[training]
epochs = 10
""")
    best_ckpt = tmp_path / "best_coco_AP_epoch_50.pth"
    best_ckpt.touch()
    
    engine = MockEngine(model=config_toml)
    assert engine.checkpoint_path == best_ckpt

def test_init_custom_toml_missing_checkpoint(tmp_path):
    """Test that missing checkpoint in TOML directory raises error during prediction, not init."""
    config_toml = tmp_path / "custom.toml"
    config_toml.write_text("""
[model]
name = "rtmdet_tiny"
num_classes = 10

[data]
root = "data/coco"

[training]
epochs = 10
""")
    
    # Init should pass with a warning
    engine = MockEngine(model=config_toml)
    assert engine.checkpoint_path is None
    
    # Predict should fail
    with pytest.raises(ValueError, match="No checkpoint found or provided"):
        engine.predict("test.jpg")

def test_init_logging_configuration():
    """Test that logging is configured during initialization."""
    with patch("loguru.logger.remove") as mock_remove:
        with patch("loguru.logger.add") as mock_add:
            MockEngine(model=ModelName.RTM_DET_TINY, num_classes=80)
            assert mock_remove.called
            assert mock_add.called