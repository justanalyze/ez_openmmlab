import pytest
from pathlib import Path
from ez_openmmlab.core.base import EZMMLab
from ez_openmmlab.schemas.model import ModelName
from typing import List
from ez_openmmlab.core.results import InferenceResult

class ConcreteEZMMLab(EZMMLab):
    """Concrete implementation for testing abstract EZMMLab."""
    def predict(self, *args, **kwargs) -> List[InferenceResult]:
        return []
    
    def _configure_model_specifics(self, config):
        pass

def test_ezmmlab_init_with_model_name_enum():
    """Test initialization with ModelName enum."""
    model = ConcreteEZMMLab(model_name=ModelName.RTM_DET_TINY)
    assert model.model_name == ModelName.RTM_DET_TINY.value

def test_ezmmlab_init_with_string():
    """Test initialization with string model name."""
    model = ConcreteEZMMLab(model_name="rtmdet_tiny")
    assert model.model_name == "rtmdet_tiny"

def test_ezmmlab_init_with_config_path(tmp_path):
    """Test initialization with a path to a config file."""
    config_file = tmp_path / "config.toml"
    config_file.touch()
    
    # This should currently fail or behave unexpectedly because the type hint 
    # and logic in base.py usually expect a ModelName or string that maps to one.
    # We want to support passing a Path object or a string path.
    
    # NOTE: The current implementation of EZMMLab.__init__ expects model_name to be ModelName | str.
    # It converts ModelName to value, but if it's a path string, it treats it as a model name.
    # However, ensure_model_checkpoint will try to resolve it.
    
    # We are testing that we CAN pass it. The logic inside might fail later (in ensure_model_checkpoint),
    # but the constructor itself should accept the type.
    
    model = ConcreteEZMMLab(model_name=config_file)
    # Ideally, we want to store the config path if it's a file, or extract the model name from it later.
    # For now, let's just assert the attribute is set.
    assert model.model_name == str(config_file)
