# Implementation Plan: Improve Engine Readability

## Phase 1: Base Class Enhancements [checkpoint: 2a7b9c1]

- [x] Task: Implement Base Helpers
    - [x] Add `_resolve_out_dir` to `EZMMLab`.
    - [x] Add `_normalize_inputs` to `EZMMLab`.
    - [x] Add tests for these new helper methods in `tests/unit/test_base_helpers.py`.
- [x] Task: Conductor - User Manual Verification 'Base Class Enhancements' (Protocol in workflow.md)

## Phase 2: Engine Refactor [checkpoint: 123456]

- [x] Task: Refactor Detection Engine
    - [x] Update `EZMMDetector.predict` to use base helpers.
    - [x] Extract `_map_inference_results` private method.
- [x] Task: Refactor Pose Engine
    - [x] Update `EZMMPose._execute_mmpose_inferencer` to use base helpers.
    - [x] Extract `_map_pose_results` private method.
- [x] Task: Verification
    - [x] Run full test suite to ensure no regressions.
- [x] Task: Conductor - User Manual Verification 'Engine Refactor' (Protocol in workflow.md)
- [ ] Task: Conductor - User Manual Verification 'Engine Refactor' (Protocol in workflow.md)
