# Specification: Lazy Result Formatting

## Overview
This track implements lazy initialization for inference results in the `InferenceResult` class. Currently, model outputs are formatted into `Boxes`, `Keypoints`, and `Masks` objects immediately upon result creation. This implementation will defer that processing until the data is actually accessed by the user, improving performance for large batches or unused result types.

## Functional Requirements
- **Lazy Properties:** `InferenceResult.boxes`, `InferenceResult.keypoints`, and `InferenceResult.masks` must be converted to properties that trigger formatting on first access.
- **Transparent Execution:** Accessing these properties must be indistinguishable from accessing pre-computed attributes (Aero-style lazy loading).
- **Decoupled Logic:** The formatting logic must reside in the `ResultFormatter` (or a similar service) and be provided to `InferenceResult` as a callback or reference.
- **Raw Data Storage:** `InferenceResult` must store the raw model output until processing is required.
- **Caching:** Once formatted, the results must be cached within the `InferenceResult` instance to avoid redundant computations.

## Non-Functional Requirements
- **Performance:** Reduced overhead during `InferenceEngine.predict()` execution.
- **Memory Efficiency:** Avoid creating complex numpy arrays/objects for result types that are never used.

## Acceptance Criteria
- [ ] `InferenceResult` initialization accepts raw data and a formatting callback.
- [ ] Accessing `.boxes` for the first time triggers the callback and returns a `Boxes` object.
- [ ] Accessing `.masks` for the first time triggers the callback and returns a `Masks` object.
- [ ] Subsequent accesses to the same property return the cached object without re-executing formatting logic.
- [ ] Existing tests for `InferenceResult` and `ResultFormatter` pass or are updated to reflect the new lazy behavior.

## Out of Scope
- Refactoring the internal data structures of `Boxes`, `Keypoints`, or `Masks`.
- Implementing lazy loading for the original image (already exists).
