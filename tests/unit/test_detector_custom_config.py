import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from ez_openmmlab import RTMDet
from ez_openmmlab.schemas.model import ModelName
import numpy as np

@patch("ez_openmmlab.engines.mmdet.DetInferencer")
@patch("ez_openmmlab.core.base.load_user_config")
@patch("ez_openmmlab.core.base.get_config_file")
@patch("ez_openmmlab.core.base.ensure_model_checkpoint")
def test_rtmdet_init_with_custom_config(mock_ensure, mock_get_config, mock_load_config, mock_inferencer_cls, tmp_path):
    """Test that RTMDet correctly initializes using a custom config.toml."""
    config_file = tmp_path / "config.toml"
    config_file.touch()
    checkpoint_file = tmp_path / "custom.pth"
    checkpoint_file.touch()
    
    # Setup mocks
    mock_user_config = MagicMock()
    mock_user_config.model.name = ModelName.RTM_DET_TINY
    mock_user_config.model.num_classes = 10
    mock_user_config.data.metainfo = {"classes": ["cat", "dog"]}
    mock_load_config.return_value = mock_user_config
    
    mock_get_config.return_value = Path("/abs/path/to/base_config.py")
    mock_ensure.return_value = checkpoint_file
    
    # Initialize RTMDet
    detector = RTMDet(model=config_file, checkpoint_path=checkpoint_file)
    
    # Trigger _init_inferencer
    detector._init_inferencer(device="cpu")
    
    # Verify DetInferencer was called with the TEMPORARY config file path
    mock_inferencer_cls.assert_called_once()
    args, kwargs = mock_inferencer_cls.call_args
    assert str(kwargs["model"]) == str(detector.config_path)
    assert str(kwargs["weights"]) == str(checkpoint_file)
