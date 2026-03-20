from pathlib import Path
from typing import Dict, Union
from loguru import logger
import subprocess


class DockerExportManager:
    """Manages the execution of MMDeploy via Docker containers."""

    def __init__(self, project_root: Union[str, Path]):
        self.project_root = Path(project_root).absolute()
        self.container_workdir = "/work"
        self.container_mmdeploy_root = "/root/workspace/mmdeploy"

    def to_container_path(self, host_path: Union[str, Path]) -> str:
        """Translates a local host path to its corresponding path inside the container.

        Args:
            host_path: Absolute or relative path on the host machine.

        Returns:
            The translated path string starting with container_workdir.

        Raises:
            ValueError: If the path is outside the project root.
        """
        path_str = str(host_path)
        # If the path is already pointing to internal MMDeploy configs, return as-is
        if path_str.startswith(self.container_mmdeploy_root):
            return path_str

        path = Path(host_path).absolute()

        try:
            relative = path.relative_to(self.project_root)
            return f"{self.container_workdir}/{relative}"
        except ValueError:
            raise ValueError(
                f"Path '{host_path}' is outside the project root '{self.project_root}'. "
                "For Docker-based export, please ensure your checkpoint and image "
                "are located within the project directory."
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
        # Path mappings
        c_deploy = self.to_container_path(deploy_cfg)
        c_model = self.to_container_path(model_cfg)
        c_checkpoint = self.to_container_path(checkpoint)
        c_test_img = self.to_container_path(test_img)
        c_work_dir = self.to_container_path(work_dir)

        gpu_flag = "--gpus all" if device == "cuda" else ""

        # 1. Setup Volumes - ONLY mount the project and specific packages
        import mmdet, mmpose

        volume_mounts = [f"-v {self.project_root}:{self.container_workdir}"]

        # We will point PYTHONPATH to this directory
        container_pkg_root = "/opt/external_pkgs"

        for pkg in [mmdet, mmpose]:
            host_pkg_path = Path(pkg.__file__).parent
            pkg_name = host_pkg_path.name
            # Important: Mount the package folder INTO the root
            # e.g., /home/.../mmdet -> /opt/external_pkgs/mmdet
            volume_mounts.append(f"-v {host_pkg_path}:{container_pkg_root}/{pkg_name}")

        # 2. Prepare the Environment Variable
        # Note: We include container_workdir so it can find local configs/models
        # and container_pkg_root so it can find 'import mmdet'
        python_path_env = (
            f"PYTHONPATH={self.container_workdir}:{container_pkg_root}:$PYTHONPATH"
        )

        deploy_script = f"{self.container_mmdeploy_root}/tools/deploy.py"
        packages = "pycocotools terminaltables shapely scipy albumentations"

        # 3. Construct the inner shell command
        # Using 'export' ensures all subsequent python calls see the new PATH
        inner_cmd = (
            f"export {python_path_env} && "
            f"pip install --no-cache-dir {packages} && "
            f"python3 {deploy_script} {c_deploy} {c_model} {c_checkpoint} {c_test_img} "
            f"--work-dir {c_work_dir} --device {device} --dump-info"
        )

        # 4. Final Docker Command
        cmd = (
            f"docker run --rm {gpu_flag} "
            f"{' '.join(volume_mounts)} "
            f"-w {self.container_workdir} "
            f"openmmlab/mmdeploy:{image_tag} "
            f"bash -c '{inner_cmd}'"
        )

        return cmd

    def check_docker_installed(self) -> bool:
        """Checks if Docker is installed and running."""
        try:
            subprocess.run(
                ["docker", "--version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def ensure_image_available(self, image_tag: str) -> None:
        """Checks for the image and interactively prompts to pull if missing."""
        if not self.check_docker_installed():
            raise RuntimeError(
                "Docker is not installed or not in your PATH.\n"
                "Model export requires Docker to run the MMDeploy container."
            )

        full_image = f"openmmlab/mmdeploy:{image_tag}"
        if self.check_image_exists(image_tag):
            return

        print(f"\n[!] The MMDeploy Docker image '{full_image}' is missing locally.")
        print(f"[!] WARNING: This image is VERY LARGE (30GB+).")
        print(
            "[!] It contains the full environment needed to export models to ONNX/TensorRT."
        )

        # Interactive prompt
        try:
            response = input(f"Do you want to download it now? (y/n): ").strip().lower()
        except EOFError:
            response = "n"  # Handle non-interactive environments

        if response != "y":
            raise RuntimeError("Export cancelled. Docker image is required.")

        logger.info(f"Pulling {full_image}... This may take a while.")
        try:
            subprocess.run(["docker", "pull", full_image], check=True)
            logger.info("Image pulled successfully.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to pull Docker image: {e}")

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

    def check_image_exists(self, image_tag: str) -> bool:
        """Checks if the specified MMDeploy image exists locally."""
        full_image = f"openmmlab/mmdeploy:{image_tag}"
        try:
            result = subprocess.run(
                ["docker", "images", "-q", full_image],
                capture_output=True,
                text=True,
                check=True,
            )
            return len(result.stdout.strip()) > 0
        except Exception:
            return False
