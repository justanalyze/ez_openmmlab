# Implementation Plan: MMDeploy Docker-based Model Export

## Phase 1: Foundation & Registry
- [ ] Task: Create deployment core structure
    - [ ] Create directory `src/ez_openmmlab/core/deploy/`
    - [ ] Initialize `src/ez_openmmlab/core/deploy/__init__.py`
- [ ] Task: Implement `DeployConfigRegistry`
    - [ ] Define internal mapping between model families (`mmdet`, `mmpose`) and MMDeploy configuration paths.
    - [ ] Implement a resolver to fetch the correct `deploy_cfg` based on format and model.
- [ ] Task: Write tests for `DeployConfigRegistry`
    - [ ] Verify correct config resolution for `mmdet` (onnx/tensorrt).
    - [ ] Verify correct config resolution for `mmpose` (onnx/tensorrt).
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Foundation & Registry' (Protocol in workflow.md)

## Phase 2: Docker Orchestration logic
- [ ] Task: Implement `DockerExportManager`
    - [ ] Create a class to handle path translation (Host path -> Container path).
    - [ ] Implement logic to construct the `docker run` command string with appropriate volume mounts.
    - [ ] Add a method to execute the command using `subprocess.run` and stream logs.
- [ ] Task: Write tests for `DockerExportManager`
    - [ ] Test path translation utility.
    - [ ] Test command string construction (mocking the filesystem).
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Docker Orchestration logic' (Protocol in workflow.md)

## Phase 3: Engine Integration
- [ ] Task: Update `EZMMLab` base class
    - [ ] Add the public `export()` method signature to the abstract base.
    - [ ] Implement the common orchestration logic in `EZMMLab.export`.
- [ ] Task: Write integration tests for `model.export()`
    - [ ] Mock the `DockerExportManager` to verify the base class calls it with correct parameters.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Engine Integration' (Protocol in workflow.md)

## Phase 4: CLI & Final Polish
- [ ] Task: Add `export` command to CLI
    - [ ] Update `src/ez_openmmlab/cli/__init__.py` to include the `export` command.
    - [ ] Map CLI arguments to the `model.export()` call.
- [ ] Task: Documentation update
    - [ ] Add "Model Export" section to `README.md` with usage examples.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: CLI & Final Polish' (Protocol in workflow.md)
