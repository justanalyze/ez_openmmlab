# Implementation Plan: Dynamic Session Dataset Registration

This plan implements a runtime registration system that dynamically creates and registers dataset classes based on `dataset.toml` metadata. This solves the "Evaluation Mismatch" problem by making custom datasets first-class citizens in the OpenMMLab registry.

## Phase 1: Registration Utility
- [x] Task: Create `src/ez_openmmlab/core/datasets.py`
    - [x] Implement `DynamicDatasetRegistry` class
    - [x] Add logic to dynamically create a Python class using `type()` inheriting from `BaseCocoStyleDataset` or `CocoDataset`
    - [x] Implement `register_module()` calls for both MMDet and MMPose registries
    - [x] Add strict collision checking to prevent overwriting existing registrations
- [x] Task: Conductor - User Manual Verification 'Phase 1: Registration Utility' (Protocol in workflow.md)

## Phase 2: Engine Integration
- [x] Task: Update `EZMMLab` workflow in `src/ez_openmmlab/core/engines/engine_base.py`
    - [x] Add a hook or call to the registry during the training initialization
    - [x] Ensure `dataset_name` is correctly extracted from the `UserConfig`
- [x] Task: Update Config Injection logic
    - [x] Modify `_run_training_workflow` to update the dictionary `type` fields in `train_dataloader` and `val_dataloader` to match the new dynamic class name
- [x] Task: Conductor - User Manual Verification 'Phase 2: Engine Integration' (Protocol in workflow.md)

## Phase 3: Verification & Testing
- [x] Task: Write Unit Tests for Dynamic Registration
    - [x] Create `tests/unit/test_dynamic_registration.py`
    - [x] Verify that a class created at runtime appears in the OpenMMLab registry
    - [x] Verify that a duplicate registration raises `ValueError`
- [x] Task: Integration Test
    - [x] Update or add a smoke test that uses a custom `dataset_name` and verifies the engine uses the new class type
- [x] Task: Conductor - User Manual Verification 'Phase 3: Verification & Testing' (Protocol in workflow.md)

## Phase 4: Quality Gate
- [x] Task: Final Code Review and Formatting
    - [x] Run `uv run --extra cpu ruff check . --fix`
    - [x] Run `uv run --extra cpu mypy .`
- [x] Task: Conductor - User Manual Verification 'Phase 4: Quality Gate' (Protocol in workflow.md)
