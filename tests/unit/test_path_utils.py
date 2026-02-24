import platform
from unittest.mock import patch
from pathlib import Path
from ez_openmmlab.core.utils.path import get_unique_dir, get_user_cache_dir


def test_get_user_cache_dir_linux():
    """Verify linux cache path convention."""
    with patch("platform.system", return_value="Linux"):
        cache_dir = get_user_cache_dir()
        assert ".cache" in str(cache_dir)
        assert "ez_openmmlab" in str(cache_dir)


def test_get_user_cache_dir_windows():
    """Verify windows cache path convention."""
    with patch("platform.system", return_value="Windows"):
        with patch.dict("os.environ", {"LOCALAPPDATA": "C:\\Users\\Test\\AppData\\Local"}):
            with patch("os.path.isdir", return_value=True):
                cache_dir = get_user_cache_dir()
                path_str = str(cache_dir).lower()
                assert "appdata" in path_str
                assert "local" in path_str
                assert "ez_openmmlab" in path_str
                assert "cache" in path_str


def test_get_unique_dir_returns_base_if_not_exists(tmp_path):
    """Verifies that the base path is returned if it doesn't exist."""
    base = tmp_path / "new_dir"
    unique = get_unique_dir(base)
    assert unique == base


def test_get_unique_dir_increments_if_exists(tmp_path):
    """Verifies that a numbered suffix is added if the path exists."""
    base = tmp_path / "preds"
    base.mkdir()

    unique = get_unique_dir(base)
    assert unique == tmp_path / "preds_1"


def test_get_unique_dir_skips_existing_increments(tmp_path):
    """Verifies that it finds the first available increment."""
    base = tmp_path / "results"
    base.mkdir()
    (tmp_path / "results_1").mkdir()
    (tmp_path / "results_2").mkdir()

    unique = get_unique_dir(base)
    assert unique == tmp_path / "results_3"
