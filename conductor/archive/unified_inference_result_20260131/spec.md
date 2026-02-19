# Track Specification: Implement Unified NumPy-First InferenceResult

## Overview
This track implements a new, standardized data structure for inference results, heavily inspired by the Ultralytics `Results` API but optimized for the OpenMMLab ecosystem. The goal is to provide a "Vectorized-First" interaction model that replaces the current list-of-objects pattern with high-performance NumPy arrays.

## Objectives
- Standardize inference output across all tasks (Detection, Pose, Segmentation).
- Improve performance by leveraging NumPy's vectorized operations instead of Python loops.
- Simplify the API for "EZ" users while providing advanced data access for power users.
- Enable self-contained visualization by storing the original image and class names within the result object.

## Functional Requirements
- **Unified Container (`InferenceResult`):**
    - Store `orig_img` (np.ndarray), `orig_shape` (tuple), `path` (str), and `names` (dict).
    - Store task-specific objects: `boxes` (Boxes), `keypoints` (Keypoints), and `masks` (Masks).
    - Include a `speed` dictionary for profiling.
- **Vectorized Helper Classes:**
    - **`Boxes`:** Manage `[N, 6]` array (x1, y1, x2, y2, score, class). Provide properties like `.xyxy`, `.conf`, and `.cls`.
    - **`Keypoints`:** Manage `[N, K, 3]` array (x, y, score). Provide properties like `.xy` and `.conf`.
    - **`Masks`:** Manage `[N, H, W]` boolean arrays. Provide properties for polygon segments.
- **Engine Integration:**
    - Update `EZMMDetector` and `EZMMPose` to return the new `InferenceResult`.
    - Ensure both single-image and batch inference (directory/list) return these standardized objects.

## Non-Functional Requirements
- **Performance:** Masking or filtering 300+ detections should be near-instant using NumPy boolean indexing.
- **API Compatibility:** The new API should feel familiar to users of `ultralytics`, making migration to `ez_openmmlab` intuitive.
- **Type Safety:** Use strict typing for all properties.

## Acceptance Criteria
- A single `InferenceResult` class handles output from RTMDet (boxes/masks) and RTMPose/RTMO (keypoints).
- Users can filter results using vectorized syntax (e.g., `result.boxes[result.boxes.conf > 0.5]`).
- The `result.plot()` or `result.show()` methods work without requiring the user to pass the original image separately.
- All existing tests pass or are updated to use the new API.
