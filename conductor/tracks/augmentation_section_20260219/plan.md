# Implementation Plan: Dedicated Augmentation Configuration and API

This plan outlines the steps to refactor the augmentation configuration into a dedicated section, enhance the training API with strict validation, and clean up the configuration injectors.

## Phase 1: Configuration Schema and Data Integrity [checkpoint: Phase 1 complete]
Focus: Define the new `AugmentationSection` and ensure it integrates correctly with `UserConfig` and TOML serialization.

- [x] Task: TDD - Create unit tests for `AugmentationSection` validation and `UserConfig` serialization in `tests/unit/test_toml_config.py`.
- [x] Task: Implement `AugmentationSection` Pydantic model in `src/ez_openmmlab/utils/toml_config.py`.
- [x] Task: Update `UserConfig` to include the `augments` field and migrate fields from `TrainingSection`.
- [x] Task: Update `load_user_config` and `save_user_config` to support the new `[augments]` section.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Configuration Schema and Data Integrity' (Protocol in workflow.md)

## Phase 2: Configuration Management and Mapping
Focus: Update the `ConfigManager` to handle the new schema and map API inputs correctly.

- [ ] Task: TDD - Update `tests/unit/test_config_manager.py` to verify mapping of augmentation parameters to the new section.
- [ ] Task: Update `ConfigManager.create_fresh_config` to populate `AugmentationSection` from training arguments.
- [ ] Task: Ensure `ConfigManager.load_metadata_from_toml` correctly extracts augmentation parameters.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Configuration Management and Mapping' (Protocol in workflow.md)

## Phase 3: Injector and Surgery Logic Refinement
Focus: Clean up `MMDetInjector` and `MMPoseInjector` to consume parameters exclusively from the new structure.

- [ ] Task: TDD - Update `tests/unit/test_injectors.py` and `tests/unit/test_pose_injector_refinement.py` to reflect the schema changes.
- [ ] Task: Refactor `MMDetInjector._patch_pipelines` to use `user_config.augments`.
- [ ] Task: Refactor `MMPoseInjector._patch_pipelines` to use `user_config.augments`.
- [ ] Task: Verify that `PipelineTransformPatcherRegistry` correctly services the refactored injectors.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Injector and Surgery Logic Refinement' (Protocol in workflow.md)

## Phase 4: API Enhancement and Runtime Validation
Focus: Implement the unified `augments` argument, strict validation, and dynamic docstrings.

- [ ] Task: TDD - Create tests for `EZMMLab.train` validation logic (unsupported keys) and docstring presence.
- [ ] Task: Update `EZMMLab.train` signature to accept `augments: Optional[Dict[str, Any]]`.
- [ ] Task: Implement strict key validation in `train()` against `PipelineTransformPatcherRegistry`.
- [ ] Task: Implement dynamic docstring generation for `train()` to list available augmentations.
- [ ] Task: Update concrete models (`RTMDet`, `RTMPose`, `RTMO`) to use the new `train` signature.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: API Enhancement and Runtime Validation' (Protocol in workflow.md)

## Phase 5: Final Integration and Regression
Focus: Ensure the entire system works end-to-end and documentation is accurate.

- [ ] Task: Run full suite of unit and integration tests.
- [ ] Task: Verify the new `[augments]` section works correctly in a real `dataset.toml` or `user_config.toml`.
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Final Integration and Regression' (Protocol in workflow.md)
