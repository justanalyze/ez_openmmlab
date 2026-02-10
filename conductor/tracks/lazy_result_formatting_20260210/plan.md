# Implementation Plan: Lazy Result Formatting

This plan implements lazy initialization for `InferenceResult` attributes (`boxes`, `keypoints`, `masks`) to improve performance by deferring raw model output processing until first access.

## Phase 1: Core Refactoring
- [x] Task: Update `InferenceResult` in `src/ez_openmmlab/core/results.py` to support lazy loading
    - [x] Modify `__init__` to accept raw data and an optional formatting callback
    - [x] Convert `boxes`, `keypoints`, and `masks` into properties
    - [x] Implement internal caching for processed results
- [x] Task: Update `ResultFormatter` interface in `src/ez_openmmlab/core/formatters.py`
    - [x] Define the interface/protocol for the lazy formatting callback
    - [x] Update `DetectionResultFormatter` to pass raw data and callback to `InferenceResult`
    - [x] Update `PoseResultFormatter` to pass raw data and callback to `InferenceResult`
- [x] Task: Conductor - User Manual Verification 'Phase 1: Core Refactoring' (Protocol in workflow.md)

## Phase 2: Verification & Testing
- [x] Task: Update Unit Tests
    - [x] Update `tests/unit/test_results.py` to verify lazy behavior (checking that formatting only happens on access)
    - [x] Ensure existing tests in `tests/unit/test_formatters.py` still pass with the new architecture
- [x] Task: Integration Testing
    - [x] Verify end-to-end inference flow in `tests/integration/test_engine_integration.py`
- [x] Task: Conductor - User Manual Verification 'Phase 2: Verification & Testing' (Protocol in workflow.md)

## Phase 3: Cleanup
- [x] Task: Final Code Review and Formatting
    - [x] Run `ruff check .` and `ruff format .`
    - [x] Run `mypy .` to ensure type safety with the new property types
- [x] Task: Conductor - User Manual Verification 'Phase 3: Cleanup' (Protocol in workflow.md)