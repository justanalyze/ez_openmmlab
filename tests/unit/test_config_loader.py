import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ez_openmmlab.core.config_loader import ConfigLoader
from ez_openmmlab.utils.download import ensure_model_checkpoint
from ez_openmmlab.schemas.model import ModelName, MODEL_URLS

@pytest.fixture
def mock_project_root(tmp_path):
    """Creates a mock project root with libs/mmdetection/configs structure."""
    mmdet_root = tmp_path / "libs" / "mmdetection" / "configs"
    mmdet_root.mkdir(parents=True)
    mmpose_root = tmp_path / "libs" / "mmpose" / "configs"
    mmpose_root.mkdir(parents=True)
    return tmp_path

def test_config_loader_init_validation(tmp_path):
    """Test that ConfigLoader validates the config root existence."""
    # Create instance without calling __init__ to manually set roots for testing
    loader = ConfigLoader.__new__(ConfigLoader)
    loader._mmdet_config_root = tmp_path / "non_existent"
    loader._mmpose_config_root = tmp_path / "non_existent"
    
    with pytest.raises(FileNotFoundError, match="Could not find local mmdetection or mmpose configs"):
        loader._validate_root()

def test_config_loader_get_config_path_success(mock_project_root):
    """Test successful config path resolution."""
    # Create a dummy config file
    config_file = mock_project_root / "libs" / "mmdetection" / "configs" / "rtmdet" / "rtmdet_tiny_8xb32-300e_coco.py"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.touch()

    with patch.object(ConfigLoader, "_validate_root", return_value=None):
        loader = ConfigLoader()
        loader._project_root = mock_project_root
        loader._mmdet_config_root = mock_project_root / "libs" / "mmdetection" / "configs"
        loader._mmpose_config_root = mock_project_root / "libs" / "mmpose" / "configs"
        
        path = loader.get_config_path("rtmdet_tiny")
        assert path.resolve() == config_file.resolve()

def test_config_loader_invalid_model():
    """Test ConfigLoader with an unsupported model name."""
    with patch.object(ConfigLoader, "_validate_root", return_value=None):
        loader = ConfigLoader()
        with pytest.raises(ValueError, match="not found in internal map"):
            loader.get_config_path("invalid_model")

def test_config_loader_missing_file(mock_project_root):
    """Test ConfigLoader when the mapped file is missing on disk."""
    with patch.object(ConfigLoader, "_validate_root", return_value=None):
        loader = ConfigLoader()
        loader._mmdet_config_root = mock_project_root / "libs" / "mmdetection" / "configs"
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
        expected_path = tmp_path / ".cache" / "ez_openmmlab" / "checkpoints" / "rtmdet_tiny.pth"
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
        assert path == tmp_path / ".cache" / "ez_openmmlab" / "checkpoints" / "unknown_model.pth"