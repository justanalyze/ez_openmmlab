# Implementation Plan: MMDeploy Docker-based Model Export

## Phase 1: Foundation & Registry [checkpoint: 75de274]
- [x] Task: Create deployment core structure [c761995]
    - [x] Create directory `src/ez_openmmlab/core/deploy/`
    - [x] Initialize `src/ez_openmmlab/core/deploy/__init__.py`
- [x] Task: Implement `DeployConfigRegistry` [12c7ffc]
    - [x] Define internal mapping between model families (`mmdet`, `mmpose`) and MMDeploy configuration paths.
    - [x] Implement a resolver to fetch the correct `deploy_cfg` based on format and model.
- [x] Task: Write tests for `DeployConfigRegistry` [12c7ffc]
    - [x] Verify correct config resolution for `mmdet` (onnx/tensorrt).
    - [x] Verify correct config resolution for `mmpose` (onnx/tensorrt).
- [x] Task: Conductor - User Manual Verification 'Phase 1: Foundation & Registry' (Protocol in workflow.md)

## Phase 2: Docker Orchestration logic [checkpoint: 39565e1]
- [x] Task: Implement `DockerExportManager` [75de274]
    - [x] Create a class to handle path translation (Host path -> Container path).
    - [x] Implement logic to construct the `docker run` command string with appropriate volume mounts.
    - [x] Add a method to execute the command using `subprocess.run` and stream logs.
- [x] Task: Write tests for `DockerExportManager` [75de274]
    - [x] Test path translation utility.
    - [x] Test command string construction (mocking the filesystem).
- [x] Task: Conductor - User Manual Verification 'Phase 2: Docker Orchestration logic' (Protocol in workflow.md)

## Phase 3: Engine Integration [checkpoint: 17aec4e]
- [x] Task: Update `EZMMLab` base class [cc78a16]
    - [x] Add the public `export()` method signature to the abstract base.
    - [x] Implement the common orchestration logic in `EZMMLab.export`.
- [x] Task: Write integration tests for `model.export()` [cc78a16]
    - [x] Mock the `DockerExportManager` to verify the base class calls it with correct parameters.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Engine Integration' (Protocol in workflow.md)

## Phase 4: CLI & Final Polish
- [ ] Task: Add `export` command to CLI
    - [ ] Update `src/ez_openmmlab/cli/__init__.py` to include the `export` command.
    - [ ] Map CLI arguments to the `model.export()` call.
- [ ] Task: Documentation update
    - [ ] Add "Model Export" section to `README.md` with usage examples.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: CLI & Final Polish' (Protocol in workflow.md)
