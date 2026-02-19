# Track Specification: Improve Engine Readability

## Overview
This track focuses on refactoring the `predict` method in `src/ez_openmmlab/engines/mmdet.py` and the `_execute_mmpose_inferencer` method in `src/ez_openmmlab/engines/mmpose.py`. These methods have become complex and difficult to read. We will improve maintainability by extracting helper methods and moving shared logic to the base class.

## Objectives
- Simplify `EZMMDetector.predict` by breaking it into smaller, composable steps.
- Simplify `EZMMPose._execute_mmpose_inferencer` by breaking it into smaller, composable steps.
- Promote code reuse by moving common logic (directory resolution, input normalization) to `EZMMLab`.

## Functional Requirements
- **Base Class Enhancements:**
    - Add `_resolve_out_dir(out_dir: Optional[str]) -> str` to `EZMMLab`.
    - Add `_normalize_inputs(inputs: Union[str, Path, list]) -> Union[str, list]` to `EZMMLab`.
- **Detection Engine Refactor:**
    - Refactor `EZMMDetector.predict` to use the new base class helpers.
    - Extract result mapping logic into a private method `_map_inference_results`.
- **Pose Engine Refactor:**
    - Refactor `EZMMPose._execute_mmpose_inferencer` to use the new base class helpers.
    - Extract result mapping logic into a private method `_map_pose_results`.

## Non-Functional Requirements
- **No Behavioral Change:** The external behavior of `predict()` must remain exactly the same.
- **Readability:** New methods must be short, well-named, and have single responsibilities.

## Acceptance Criteria
- `EZMMDetector.predict` is significantly shorter and reads as a high-level orchestration flow.
- `EZMMPose._execute_mmpose_inferencer` is significantly shorter and reads as a high-level orchestration flow.
- Shared logic resides in `EZMMLab`.
- All existing tests (unit and E2E) pass without modification.
