# Implementation Plan: Lazy Result Formatting

This plan implements lazy initialization for `InferenceResult` attributes (`boxes`, `keypoints`, `masks`) to improve performance by deferring raw model output processing until first access.

## Phase 1: Core Refactoring
- [ ] Task: Update `InferenceResult` in `src/ez_openmmlab/core/results.py` to support lazy loading
    - [ ] Modify `__init__` to accept raw data and an optional formatting callback
    - [ ] Convert `boxes`, `keypoints`, and `masks` into properties
    - [ ] Implement internal caching for processed results
- [ ] Task: Update `ResultFormatter` interface in `src/ez_openmmlab/core/formatters.py`
    - [ ] Define the interface/protocol for the lazy formatting callback
    - [ ] Update `DetectionResultFormatter` to pass raw data and callback to `InferenceResult`
    - [ ] Update `PoseResultFormatter` to pass raw data and callback to `InferenceResult`
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Core Refactoring' (Protocol in workflow.md)

## Phase 2: Verification & Testing
- [ ] Task: Update Unit Tests
    - [ ] Update `tests/unit/test_results.py` to verify lazy behavior (checking that formatting only happens on access)
    - [ ] Ensure existing tests in `tests/unit/test_formatters.py` still pass with the new architecture
- [ ] Task: Integration Testing
    - [ ] Verify end-to-end inference flow in `tests/integration/test_engine_integration.py`
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Verification & Testing' (Protocol in workflow.md)

## Phase 3: Cleanup
- [ ] Task: Final Code Review and Formatting
    - [ ] Run `ruff check .` and `ruff format .`
    - [ ] Run `mypy .` to ensure type safety with the new property types
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Cleanup' (Protocol in workflow.md)
