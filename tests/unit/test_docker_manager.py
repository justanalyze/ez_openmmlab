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

    # Internal container path (should pass through)
    internal_path = "/root/workspace/mmdeploy/configs/config.py"
    assert manager.to_container_path(internal_path) == internal_path


def test_command_construction():
    """Test that the docker command string is correctly built."""
    root = Path("/tmp/mock_project")
    manager = DockerExportManager(project_root=root)

    cmd = manager.build_command(
        deploy_cfg="/root/workspace/mmdeploy/configs/deploy.py",
        model_cfg=root / "configs/model.py",
        checkpoint=root / "weights/best.pth",
        test_img=root / "demos/test.jpg",
        work_dir=root / "runs/export",
        device="cpu",
    )

    assert "docker run --rm" in cmd
    assert f'-v "{root.as_posix()}":/work' in cmd
    assert "openmmlab/mmdeploy:latest" in cmd
    assert "python3 /root/workspace/mmdeploy/tools/deploy.py" in cmd
    assert "/root/workspace/mmdeploy/configs/deploy.py" in cmd
    assert "/work/configs/model.py" in cmd
    assert "/work/weights/best.pth" in cmd
    assert "/work/demos/test.jpg" in cmd
    assert "--work-dir /work/runs/export" in cmd
    assert "--device cpu" in cmd


def test_pythonpath_surgical_injection():
    """Verify that PYTHONPATH includes the new surgical mount point."""
    root = Path("/tmp/mock_project")
    manager = DockerExportManager(project_root=root)

    # Mock mmdet and mmpose modules
    mock_mmdet = MagicMock()
    mock_mmdet.__file__ = str(
        root / "venv/lib/python3.10/site-packages/mmdet/__init__.py"
    )

    mock_mmpose = MagicMock()
    mock_mmpose.__file__ = str(
        root / "venv/lib/python3.10/site-packages/mmpose/__init__.py"
    )

    with patch.dict("sys.modules", {"mmdet": mock_mmdet, "mmpose": mock_mmpose}):
        with (
            patch(
                "ez_openmmlab.core.deploy.docker_manager.mmdet", mock_mmdet, create=True
            ),
            patch(
                "ez_openmmlab.core.deploy.docker_manager.mmpose",
                mock_mmpose,
                create=True,
            ),
        ):
            cmd = manager.build_command(
                deploy_cfg="/root/workspace/mmdeploy/configs/deploy.py",
                model_cfg=root / "config.py",
                checkpoint=root / "best.pth",
                test_img=root / "test.jpg",
                work_dir=root / "export",
            )

            # 1. Verify Volume Mounts:
            # It should mount the HOST directory to the CONTAINER /opt/external_pkgs path
            assert (
                f'-v "{root.as_posix()}/venv/lib/python3.10/site-packages/mmdet":/opt/external_pkgs/mmdet'
                in cmd
            )
            assert (
                f'-v "{root.as_posix()}/venv/lib/python3.10/site-packages/mmpose":/opt/external_pkgs/mmpose'
                in cmd
            )

            # 2. Verify PYTHONPATH:
            # It should point to the container's /opt/external_pkgs root, NOT the host's venv path
            expected_env = "PYTHONPATH=/work:/opt/external_pkgs:$PYTHONPATH"
            assert expected_env in cmd


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
