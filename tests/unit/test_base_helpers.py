import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ez_openmmlab.core.base import EZMMLab
from ez_openmmlab.schemas.model import ModelName

class ConcreteEZMMLab(EZMMLab):
    """Concrete implementation for testing abstract base class."""
    def predict(self, *args, **kwargs):
        pass
    def _configure_model_specifics(self, config):
        pass

@pytest.fixture
def ez_instance():
    with patch("ez_openmmlab.core.base.ensure_model_checkpoint", return_value=Path("dummy.pth")):
        return ConcreteEZMMLab(ModelName.RTM_DET_TINY)

def test_resolve_out_dir(ez_instance, tmp_path):
    """Test unique directory resolution helper."""
    # Case 1: None -> Empty string
    assert ez_instance._resolve_out_dir(None) == ""
    
    # Case 2: New directory
    out_dir = tmp_path / "new_out"
    assert ez_instance._resolve_out_dir(str(out_dir)) == str(out_dir)
    
    # Case 3: Existing directory -> Incremented
    out_dir.mkdir()
    expected = tmp_path / "new_out_1"
    assert ez_instance._resolve_out_dir(str(out_dir)) == str(expected)

def test_normalize_inputs_single_file(ez_instance):
    """Test normalizing a single file path string."""
    path = "test.jpg"
    assert ez_instance._normalize_inputs(path) == "test.jpg"
    assert ez_instance._normalize_inputs(Path(path)) == "test.jpg"

def test_normalize_inputs_list(ez_instance):
    """Test normalizing a list of paths."""
    paths = ["a.jpg", Path("b.png")]
    result = ez_instance._normalize_inputs(paths)
    assert result == ["a.jpg", "b.png"]

def test_normalize_inputs_directory(ez_instance, tmp_path):
    """Test normalizing a directory path by globbing images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "1.jpg").touch()
    (img_dir / "2.PNG").touch()
    (img_dir / "text.txt").touch() # Should be ignored
    
    result = ez_instance._normalize_inputs(img_dir)
    assert isinstance(result, list)
    assert len(result) == 2
    assert any("1.jpg" in p for p in result)
    assert any("2.PNG" in p for p in result)
