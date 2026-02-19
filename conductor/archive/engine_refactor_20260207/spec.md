# Specification: Refactor Core Engine and Orchestration

## Overview
Refactor the core engine architecture (`src/ez_openmmlab/core/base.py`) and its specialized implementations (`src/ez_openmmlab/engines/**`) to improve adherence to SOLID principles, specifically the Single Responsibility Principle (SRP). The goal is to reduce "bloat" in the base class by extracting auxiliary responsibilities into dedicated utilities, standardizing configuration building, and decoupling result formatting.

## Goals
- **De-bloat `EZMMLab`:** Remove logic not directly related to orchestration and move it to helper classes.
- **Standardize Configuration Building:** Move `UserConfig` assembly and dummy config preparation into a dedicated `UserConfigBuilder`.
- **Decouple Result Formatting:** Extract result mapping and vectorized container assembly into task-specific `ResultFormatter` utilities.
- **Improve Maintainability:** Eliminate inconsistent patterns between Detection and Pose engines.

## Functional Requirements
1.  **Extract Result Formatters:**
    - Create `ResultFormatter` classes for Detection and Pose tasks.
    - Move `_map_inference_results`, `_process_single_prediction`, and vectorized data assembly into these formatters.
2.  **Centralize UserConfig Construction:**
    - Create a `UserConfigBuilder` to handle `UserConfig` assembly, validation, and the preparation of dummy configurations required for standard model inference.
3.  **Refactor `EZMMLab` Base Class:**
    - Update `EZMMLab` to delegate input normalization, resource discovery (metadata loading), and configuration building to the `UserConfigBuilder` and other specialized helpers.
    - Maintain the "Enhanced Template Method" pattern for high-level `predict` and `train` workflows.

## Non-Functional Requirements
- **API Stability:** No changes to the public `RTMDet`, `RTMPose`, or `RTMO` constructor or `predict/train` signatures.
- **Performance:** Ensure lazy loading patterns (e.g., for images) are preserved or improved.
- **Testability:** Isolated utilities should be easier to unit test than internal private methods.

## Acceptance Criteria
- `EZMMLab` lines of code reduced by at least 30%.
- No hardcoded task-specific logic (e.g., `if "rtmpose" in self.model`) inside the shared base methods.
- Existing unit and integration tests pass without modification.
- New unit tests for extracted `ResultFormatter` and `UserConfigBuilder`.

## Out of Scope
- Implementation of new model families.
- Refactoring of the CLI layer.
- Modification of underlying MMLab libraries.
