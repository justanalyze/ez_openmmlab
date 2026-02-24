import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from ez_openmmlab.core.engines.engine_base import EZMMLab
from ez_openmmlab.core.schema.models import ModelName

# Concrete implementation for testing abstract base class
class ConcreteEZMMLab(EZMMLab):
    def _init_inferencer(self, device: str, **kwargs) -> None:
        pass
    def _run_inference(self, inputs: list, out_dir: str, show: bool, **kwargs):
        return []
    def _get_library_family(self) -> str:
        return "mmdet"
    def _get_architecture_params(self, **kwargs):
        return {}

def test_ezmmlab_init_with_model_name_enum():
    """Test initialization with ModelName enum."""
    # We pass num_classes to satisfy the custom weights check in _validate_inputs
    model_obj = ConcreteEZMMLab(model=ModelName.RTM_DET_TINY, num_classes=80)
    assert model_obj.model == "rtmdet_tiny"
    assert "rtmdet_tiny" in str(model_obj.config_path)

def test_ezmmlab_init_with_string():
    """Test initialization with string model name."""
    model_obj = ConcreteEZMMLab(model="rtmdet_tiny", num_classes=80)
    assert model_obj.model == "rtmdet_tiny"

def test_ezmmlab_init_with_config_path_no_checkpoint(tmp_path):
    """Test initialization with a path to a config file but no checkpoint.
    
    Now expects a warning during init and a ValueError during predict.
    """
    config_file = tmp_path / "config.toml"
    # Create a valid minimal TOML to pass validation
    config_file.write_text("""
[model]
name = "rtmdet_tiny"
num_classes = 80

[data]
root = "data/coco"

[training]
epochs = 10
""")

    engine = ConcreteEZMMLab(model=config_file)
    assert engine.checkpoint_path is None
    
    with pytest.raises(ValueError, match="No checkpoint found or provided"):
        engine.predict("test.jpg")
