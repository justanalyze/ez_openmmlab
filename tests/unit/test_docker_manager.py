import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ez_openmmlab.core.deploy.docker_manager import DockerExportManager

def test_path_translation():
    """Test that host paths are correctly translated to container paths."""
    root = Path("/tmp/mock_project")
    manager = DockerExportManager(project_root=root)
    
    # Path inside project
    host_path = root / "weights/model.pth"
    container_path = manager.to_container_path(host_path)
    assert container_path == "/work/weights/model.pth"
    
    # Absolute path outside project (should handle gracefully or error)
    with pytest.raises(ValueError, match="outside the project root"):
        manager.to_container_path("/tmp/other.pth")

def test_command_construction():
    """Test that the docker command string is correctly built."""
    root = Path("/tmp/mock_project")
    manager = DockerExportManager(project_root=root)
    
    cmd = manager.build_command(
        deploy_cfg=root / "configs/deploy.py",
        model_cfg=root / "configs/model.py",
        checkpoint=root / "weights/best.pth",
        test_img=root / "demos/test.jpg",
        work_dir=root / "runs/export",
        device="cpu"
    )
    
    assert "docker run --rm" in cmd
    assert f"-v {root}:/work" in cmd
    assert "openmmlab/mmdeploy:latest" in cmd
    assert "python /mmdeploy/tools/deploy.py" in cmd
    assert "/work/configs/deploy.py" in cmd
    assert "/work/configs/model.py" in cmd
    assert "/work/weights/best.pth" in cmd
    assert "/work/demos/test.jpg" in cmd
    assert "--work-dir /work/runs/export" in cmd
    assert "--device cpu" in cmd

@patch("subprocess.run")
def test_run_execution(mock_run):
    """Test that the subprocess is called with the correct command."""
    manager = DockerExportManager(project_root="/home/user/project")
    mock_run.return_value = MagicMock(returncode=0)
    
    manager.run_export("docker cmd string")
    
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert "docker cmd string" in args[0]
    assert kwargs["shell"] is True
