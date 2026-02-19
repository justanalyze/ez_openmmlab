# Specification: SOLID Refactoring of Engine Inheritance

## Overview
This track focuses on applying SOLID principles (specifically SRP and DRY) to the engine hierarchy. Currently, `EZMMDetector` and `EZMMPose` duplicate the orchestration logic for inference (path normalization, result mapping, output directory handling). We will unify this using the **Template Method Pattern** in the base `EZMMLab` class.

## Goals
- **Eliminate Code Duplication**: Move shared `predict()` orchestration to `EZMMLab`.
- **Enforce Consistent API**: Ensure all engines follow the same high-level workflow.
- **Improve Maintainability**: Child classes should only handle library-specific API calls.

## Functional Requirements
1.  **Refactor `EZMMLab` (Base)**:
    - Implement `predict()` as a Template Method.
    - Implement a shared `_get_class_names()` helper.
    - Define abstract hooks: `_init_inferencer(self, device, **kwargs)` and `_run_inference(self, inputs, out_dir, show, **kwargs)`.
2.  **Refactor `EZMMDetector`**:
    - Remove `predict()` and `_get_class_names()`.
    - Implement `_init_inferencer()` for `DetInferencer`.
    - Implement `_run_inference()` to call the inferencer and handle the `confidence` -> `pred_score_thr` mapping.
3.  **Refactor `EZMMPose`**:
    - Remove `predict()` and `_get_class_names()`.
    - Implement `_run_inference()` to handle the MMPose generator and return a list of results.
    - Ensure `_init_inferencer()` remains abstract or is correctly implemented in its children (`RTMPose`, `RTMO`).

## Non-Functional Requirements
- **LSP Compliance**: All `predict()` methods must maintain binary compatibility with existing usage.
- **Robustness**: Ensure `**kwargs` are passed correctly down to the underlying inferencers.

## Acceptance Criteria
- [ ] `EZMMLab.predict` is the single source of orchestration logic.
- [ ] `EZMMDetector` and `EZMMPose` contain only library-specific logic.
- [ ] All unit and integration tests pass (Detection, Pose, Segmentation).
- [ ] Demo scripts run without modification.
