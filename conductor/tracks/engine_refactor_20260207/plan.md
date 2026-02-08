# Implementation Plan: Refactor Core Engine and Orchestration

## Phase 1: Decouple Result Formatting [checkpoint: c4bd3be]
Focus on extracting the logic that transforms raw MMLab outputs into vectorized `InferenceResult` objects.

- [x] Task: Create `ResultFormatter` abstract base and `DetectionResultFormatter`
    - [x] Define `ResultFormatter` interface in `src/ez_openmmlab/core/formatters.py`
    - [x] Implement `DetectionResultFormatter` with `_map_results` and `_process_single` logic
    - [x] Write unit tests for `DetectionResultFormatter`
- [x] Task: Create `PoseResultFormatter`
    - [x] Implement `PoseResultFormatter` handling keypoint and bbox extraction
    - [x] Write unit tests for `PoseResultFormatter`
- [x] Task: Integrate Formatters into Engines
    - [x] Update `EZMMDetector` to delegate to `DetectionResultFormatter`
    - [x] Update `EZMMPose` to delegate to `PoseResultFormatter`
    - [x] Verify with integration tests
- [x] Task: Conductor - User Manual Verification 'Phase 1: Decouple Result Formatting' (Protocol in workflow.md)

## Phase 2: Centralize UserConfig Construction
Extract the logic for building `UserConfig` and preparing the dummy configurations used for standard model inference.

- [~] Task: Create `UserConfigBuilder` utility
    - [ ] Define `UserConfigBuilder` in `src/ez_openmmlab/core/config_builder.py`
    - [ ] Move `_auto_load_metadata` and `UserConfig` assembly logic from `EZMMLab`
    - [ ] Write unit tests for `UserConfigBuilder`
- [ ] Task: Implement Dummy Configuration Lifecycle
    - [ ] Move temporary python config generation (`_resolve_model_config` logic) to `UserConfigBuilder`
    - [ ] Ensure proper cleanup/deletion of temporary files
    - [ ] Write unit tests for config lifecycle
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Centralize UserConfig Construction' (Protocol in workflow.md)

## Phase 3: Refactor EZMMLab Base Class
Final cleanup of `EZMMLab` to focus purely on high-level orchestration.

- [ ] Task: Move Auxiliary Helpers to Utilities
    - [ ] Relocate `_normalize_inputs` and image globbing to a specialized helper
    - [ ] Move `_resolve_out_dir` to path utilities
    - [ ] Update all references
- [ ] Task: Refactor EZMMLab to Delegate to specialized components
    - [ ] Update `EZMMLab.__init__` to use `UserConfigBuilder`
    - [ ] Simplify `train` and `predict` by removing boilerplate
    - [ ] Ensure "Template Method" pattern is clean and readable
- [ ] Task: Verify SOLID Adherence and Stability
    - [ ] Perform a final audit of class responsibilities
    - [ ] Run full test suite (unit + integration) to ensure no regressions
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Refactor EZMMLab Base Class' (Protocol in workflow.md)