# Specification: MMDeploy Docker-based Model Export

## Overview
This track introduces a unified `.export()` method to the `ez_openmmlab` engine. By leveraging MMDeploy via Docker, it allows users to convert trained or pre-trained models into production-ready formats (ONNX, TensorRT) without dealing with complex local dependency installations.

## Functional Requirements
- **Unified Export API**: Implement `EZMMLab.export(format, image, output_dir, **kwargs)` in the base class.
- **Support for Key Formats**:
    - `onnx`: Export to standard ONNX format using ONNX Runtime configs.
    - `tensorrt`: Export to NVIDIA TensorRT engines.
- **Docker Orchestration**:
    - Automatically build and execute `docker run` commands targeting `openmmlab/mmdeploy:latest`.
    - Handle volume mounting to ensure the container can see the model weights, configs (including vendored libs), and the sample image.
    - Implement path translation between the host filesystem and the container's internal `/work` directory.
- **Deployment Config Mapping**:
    - Create an internal registry that maps `ez_openmmlab` model families (e.g., `mmdet`, `mmpose`) to the appropriate MMDeploy `deploy_cfg` files.
- **User Inputs**:
    - Require a sample image from the user to trace the model graph.
    - Allow customization of the output directory for artifacts.

## Non-Functional Requirements
- **Simplicity ("EZ")**: The user should not need to know any Docker syntax.
- **Robustness**: Provide clear error messages if Docker is not installed or the image cannot be pulled.
- **Path Safety**: Ensure absolute paths are used for all mounts to prevent "file not found" errors inside the container.

## Acceptance Criteria
- Calling `model.export(format='onnx', image='sample.jpg')` produces a valid `.onnx` file in the specified directory.
- Calling `model.export(format='tensorrt', image='sample.jpg')` produces a valid TensorRT engine.
- The system correctly handles models from both `mmdet` and `mmpose` families.
- The process cleans up the Docker container (`--rm`) after execution.

## Out of Scope
- Support for other formats (OpenVINO, CoreML) in the initial version.
- Custom MMDeploy build support (sticking to official Docker images).
- Local (non-Docker) MMDeploy execution.
