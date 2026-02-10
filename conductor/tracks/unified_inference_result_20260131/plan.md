# Implementation Plan: Implement Unified NumPy-First InferenceResult

## Phase 1: Core Data Structures [checkpoint: 8a9b0c1]

- [x] Task: Implement Vectorized Helper Classes
  - [x] Create `src/ez_openmmlab.core.inference.results.py`.
  - [x] Write failing tests for `BaseData`, `Boxes`, `Keypoints`, and `Masks` in `tests/unit/test_results.py`.
  - [x] Implement `BaseData` class with indexing and length support.
  - [x] Implement `Boxes` class with `.xyxy`, `.conf`, and `.cls` properties.
  - [x] Implement `Keypoints` class with `.xy` and `.conf` properties.
  - [x] Implement `Masks` class with support for boolean masks and polygon extraction.
  - [x] Run tests and verify they pass (Green Phase).
- [x] Task: Implement Main Container Class
  - [x] Write failing tests for `InferenceResult` container in `tests/unit/test_results.py`.
  - [x] Implement `InferenceResult` with attributes for `orig_img`, `boxes`, `keypoints`, `masks`, `speed`, and `names`.
  - [x] Implement `InferenceResult.plot()` and `InferenceResult.show()` methods.
  - [x] Run tests and verify they pass (Green Phase).
- [x] Task: Conductor - User Manual Verification 'Core Data Structures' (Protocol in workflow.md)

## Phase 2: Engine Integration [checkpoint: 9b0c1d2]

- [x] Task: Update Detection Engine
  - [x] Write integration tests in `tests/integration/test_result_migration.py` verifying `EZMMDetector` returns new `InferenceResult`.
  - [x] Update `EZMMDetector._map_inference_results` to instantiate and return the new `InferenceResult`.
  - [x] Verify detection demos and existing tests pass.
- [x] Task: Update Pose Engine
  - [x] Write integration tests in `tests/integration/test_result_migration.py` verifying `EZMMPose` returns new `InferenceResult`.
  - [x] Update `EZMMPose._map_pose_results` to instantiate and return the new `InferenceResult`.
  - [x] Verify pose demos and existing tests pass.
- [x] Task: Conductor - User Manual Verification 'Engine Integration' (Protocol in workflow.md)

## Phase 3: Cleanup and Finalization [checkpoint: a1b2c3d]

- [x] Task: Migration and Deprecation
  - [x] Search codebase for usages of old `InferenceResult` and `PoseInferenceResult` in `src/ez_openmmlab/schemas/inference.py`.
  - [x] Update all CLI commands and demos to use the new vectorized API.
  - [x] Deprecate or remove `src/ez_openmmlab/schemas/inference.py` if no longer used.
- [x] Task: Documentation and Quality
  - [x] Update docstrings and type hints throughout the project.
  - [x] Run full test suite and verify >80% coverage.
  - [x] Perform final linting and formatting check with Ruff.
- [x] Task: Conductor - User Manual Verification 'Cleanup and Finalization' (Protocol in workflow.md)
