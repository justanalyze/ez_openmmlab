import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from ez_openmmlab.core.base import EZMMLab
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.toml_config import UserConfig
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

def test_ezmmlab_init_with_config_path_no_checkpoint(tmp_path):
    """Test initialization with a path to a config file but no checkpoint raises ValueError."""
    config_file = tmp_path / "config.toml"
    config_file.touch()
    
    with pytest.raises(ValueError, match="Checkpoint path is required"):
        ConcreteEZMMLab(model_name=config_file)

@patch("ez_openmmlab.core.base.get_config_file")
@patch("ez_openmmlab.core.base.load_user_config")
def test_resolve_model_config_with_toml_valid(mock_load_config, mock_get_config, tmp_path):
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
    
    # Mock base config path
    mock_get_config.return_value = Path("/libs/mmdetection/configs/rtmdet/tiny.py")
    
    # We also need to mock ensure_model_checkpoint because __init__ calls it.
    # But since we provide a checkpoint_path, it might just resolve it.
    # However, ensure_model_checkpoint logic for custom models might be tricky.
    # In base.py:
    # if isinstance(model_name, (Path, str)) and str(model_name).endswith(".toml"):
    #      self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    # else:
    #      self.checkpoint_path = ensure_model_checkpoint(...)
    
    # So if we pass a .toml, ensure_model_checkpoint is skipped. Good.
    
    model = ConcreteEZMMLab(model_name=config_file, checkpoint_path=checkpoint_file)
    
    # Check if temp config was set
    assert model._temp_config_file is not None
    assert model._temp_config_file.exists()
    assert model._temp_config_file.suffix == ".py"
    
    # Check content
    content = model._temp_config_file.read_text()
    assert '_base_ = ["/libs/mmdetection/configs/rtmdet/tiny.py"]' in content
    
    # Verify model state was updated from config
    assert model.model_name == ModelName.RTM_DET_TINY.value
    assert model.num_classes == 80

def test_temp_config_cleanup(tmp_path):
    """Test that temporary config file is deleted when object is destroyed."""
    config_file = tmp_path / "config.toml"
    config_file.touch()
    checkpoint_file = tmp_path / "epoch_10.pth"
    checkpoint_file.touch()
    
    with patch("ez_openmmlab.core.base.load_user_config") as mock_load:
        mock_user_config = MagicMock()
        mock_user_config.model.name = ModelName.RTM_DET_TINY
        mock_load.return_value = mock_user_config
        
        with patch("ez_openmmlab.core.base.get_config_file") as mock_get:
            mock_get.return_value = Path("/dummy.py")
            
            model = ConcreteEZMMLab(model_name=config_file, checkpoint_path=checkpoint_file)
            temp_path = model._temp_config_file
            assert temp_path.exists()
            
            # Destroy object
            del model
            
            # File should be gone
            assert not temp_path.exists()
