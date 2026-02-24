from pathlib import Path

from ez_openmmlab.core.utils.input import normalize_inputs
from ez_openmmlab.core.utils.path import get_unique_dir


def test_resolve_out_dir(tmp_path):
    """Test unique directory resolution utility."""
    # Logic: if provided, resolve to unique. If not, handled by engine (empty string)

    # Case 1: New directory
    out_dir = tmp_path / "new_out"
    assert get_unique_dir(out_dir) == out_dir

    # Case 2: Existing directory -> Incremented
    out_dir.mkdir()
    expected = tmp_path / "new_out_1"
    assert get_unique_dir(out_dir) == expected


def test_normalize_inputs_single_file():
    """Test normalizing a single file path string."""
    path = "test.jpg"
    assert normalize_inputs(path) == "test.jpg"
    assert normalize_inputs(Path(path)) == "test.jpg"


def test_normalize_inputs_list():
    """Test normalizing a list of paths."""
    paths = ["a.jpg", Path("b.png")]
    result = normalize_inputs(paths)
    assert result == ["a.jpg", "b.png"]


def test_normalize_inputs_directory(tmp_path):
    """Test normalizing a directory path by globbing images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "1.jpg").touch()
    (img_dir / "2.PNG").touch()
    (img_dir / "text.txt").touch() # Should be ignored

    result = normalize_inputs(img_dir)
    assert isinstance(result, list)
    assert len(result) == 2
    assert any("1.jpg" in p for p in result)
    assert any("2.PNG" in p for p in result)