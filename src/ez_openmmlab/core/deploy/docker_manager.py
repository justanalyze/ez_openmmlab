from pathlib import Path
from typing import Union
from loguru import logger
import subprocess


class DockerExportManager:
    """Manages the execution of MMDeploy via Docker containers."""

    def __init__(self, project_root: Union[str, Path]):
        self.project_root = Path(project_root).absolute()
        self.container_workdir = "/work"

    def to_container_path(self, host_path: Union[str, Path]) -> str:
        """Translates a local host path to its corresponding path inside the container.

        Args:
            host_path: Absolute or relative path on the host machine.

        Returns:
            The translated path string starting with container_workdir.

        Raises:
            ValueError: If the path is outside the project root and cannot be mapped.
        """
        path = Path(host_path).absolute()
        try:
            relative = path.relative_to(self.project_root)
            return f"{self.container_workdir}/{relative}"
        except ValueError:
            raise ValueError(
                f"Path '{host_path}' is outside the project root '{self.project_root}' "
                "and cannot be automatically mapped to the Docker container volumes."
            )

    def build_command(
        self,
        deploy_cfg: str,
        model_cfg: str,
        checkpoint: str,
        test_img: str,
        work_dir: str,
        device: str = "cpu",
        image_tag: str = "latest",
    ) -> str:
        """Constructs the full 'docker run' command string.

        Args:
            deploy_cfg: Path to deploy config (relative to project or absolute).
            model_cfg: Path to model config.
            checkpoint: Path to model weights.
            test_img: Path to sample image for tracing.
            work_dir: Host relative path for output artifacts.
            device: 'cpu' or 'cuda'.
            image_tag: The tag for 'openmmlab/mmdeploy' image.

        Returns:
            A string containing the complete docker command.
        """
        # Map all host paths to container paths
        c_deploy = self.to_container_path(deploy_cfg)
        c_model = self.to_container_path(model_cfg)
        c_checkpoint = self.to_container_path(checkpoint)
        c_test_img = self.to_container_path(test_img)
        c_work_dir = self.to_container_path(work_dir)

        gpu_flag = "--gpus all" if device == "cuda" else ""

        cmd = (
            f"docker run --rm {gpu_flag} "
            f"-v {self.project_root}:{self.container_workdir} "
            f"-w {self.container_workdir} "
            f"openmmlab/mmdeploy:{image_tag} "
            f"python /mmdeploy/tools/deploy.py "
            f"{c_deploy} {c_model} {c_checkpoint} {c_test_img} "
            f"--work-dir {c_work_dir} "
            f"--device {device}"
        )
        return cmd

    def run_export(self, command: str) -> None:
        """Executes the docker command using subprocess.

        Args:
            command: The full command string to execute.

        Raises:
            RuntimeError: If the command fails.
        """
        logger.info("Starting MMDeploy export via Docker...")
        logger.debug(f"Executing: {command}")

        try:
            # We use shell=True to handle the complex command string with flags and mounts
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                text=True,
                capture_output=False,  # Stream to stdout/stderr directly
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker export failed with exit code {e.returncode}")
            raise RuntimeError(
                "MMDeploy export failed. Ensure Docker is running and the image is available."
            ) from e
