# Implementation Plan: Refactor Core Engine and Orchestration

## Phase 1: Decouple Result Formatting [checkpoint: c4bd3be]
Focus on extracting the logic that transforms raw MMLab outputs into vectorized `InferenceResult` objects.

- [x] Task: Create `ResultFormatter` abstract base and `DetectionResultFormatter`
    - [x] Define `ResultFormatter` interface in `src/ez_openmmlab.core.inference.formatters.py`
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

## Phase 2: Centralize UserConfig Construction [checkpoint: 1b7ecf6]
Extract the logic for building `UserConfig` and preparing the dummy configurations used for standard model inference.

- [x] Task: Create `UserConfigBuilder` utility
    - [x] Define `UserConfigBuilder` in `src/ez_openmmlab/core/config_builder.py`
    - [x] Move `_auto_load_metadata` and `UserConfig` assembly logic from `EZMMLab`
    - [x] Write unit tests for `UserConfigBuilder`
- [x] Task: Implement Dummy Configuration Lifecycle
    - [x] Move temporary python config generation (`_resolve_model_config` logic) to `UserConfigBuilder`
    - [x] Ensure proper cleanup/deletion of temporary files
    - [x] Write unit tests for config lifecycle
- [x] Task: Conductor - User Manual Verification 'Phase 2: Centralize UserConfig Construction' (Protocol in workflow.md)

## Phase 3: Refactor EZMMLab Base Class [checkpoint: 668503a]
Final cleanup of `EZMMLab` to focus purely on high-level orchestration.

- [x] Task: Move Auxiliary Helpers to Utilities
    - [x] Relocate `_normalize_inputs` and image globbing to a specialized helper
    - [x] Move `_resolve_out_dir` to path utilities
    - [x] Update all references
- [x] Task: Refactor EZMMLab to Delegate to specialized components
    - [x] Update `EZMMLab.__init__` to use `UserConfigBuilder`
    - [x] Simplify `train` and `predict` by removing boilerplate
    - [x] Ensure "Template Method" pattern is clean and readable
- [x] Task: Verify SOLID Adherence and Stability
    - [x] Perform a final audit of class responsibilities
    - [x] Run full test suite (unit + integration) to ensure no regressions
- [x] Task: Conductor - User Manual Verification 'Phase 3: Refactor EZMMLab Base Class' (Protocol in workflow.md)