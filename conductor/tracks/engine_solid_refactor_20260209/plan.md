# Implementation Plan: SOLID Refactoring of Engine Inheritance

## Phase 1: Base Class Refactor and Regression Tests [checkpoint: 2a41bdc]

- [x] Task: Update `EZMMLab` with Template Method [2a41bdc]
    - [x] Modify `src/ez_openmmlab/core/engines/engine_base.py`:
        - [x] Implement `predict()`.
        - [x] Implement `_get_class_names()`.
        - [x] Add `ABC` abstract methods for hooks.
    - [x] Create `tests/unit/test_engine_solid_logic.py` using a mock engine to verify the template workflow.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Base Class Refactor' (Protocol in workflow.md) [2a41bdc]

## Phase 2: Refactor Concrete Engines [checkpoint: 2a41bdc]

- [x] Task: Refactor `EZMMDetector` [2a41bdc]
    - [x] Update `src/ez_openmmlab/core/engines/mmdet.py`:
        - [x] Remove `predict()` and `_get_class_names()`.
        - [x] Implement `_init_inferencer()` and `_run_inference()`.
- [x] Task: Refactor `EZMMPose` and Children [2a41bdc]
    - [x] Update `src/ez_openmmlab/core/engines/mmpose.py`:
        - [x] Remove `predict()` and `_get_class_names()`.
        - [x] Implement `_run_inference()`.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Refactor Concrete Engines' (Protocol in workflow.md) [2a41bdc]

## Phase 3: Final Verification [checkpoint: 2a41bdc]

- [x] Task: Verify All Engine Variants [2a41bdc]
    - [x] Run automated tests: `uv run pytest`.
    - [x] Run Detection demo: `uv run --extra cpu python demos/demo_rtmdet-bbox.py`.
    - [x] Run Pose demo: `uv run --extra cpu python demos/demo_rtmpose.py`.
    - [x] Run Segmentation demo: `uv run --extra cpu python demos/demo_rtmdet-segmentation.py`.

