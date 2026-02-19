# Specification: Dedicated Augmentation Configuration and API

## 1. Overview
Currently, augmentation parameters (like `scale_factor`, `rotate_factor`, and `random_flip_prob`) are mixed into the `TrainingSection` of the `UserConfig`. This track introduces a dedicated `AugmentationSection` to improve organization, data integrity, and API clarity. It also implements strict validation against the configuration "surgery" registry to ensure only supported augmentations are applied and provides dynamic documentation for available augmentations.

## 2. Functional Requirements

### 2.1. Configuration Schema (`UserConfig`)
- Define a new `AugmentationSection` Pydantic model in `src/ez_openmmlab/utils/toml_config.py`.
- Move/Consolidate augmentation fields into `AugmentationSection`:
    - `scale_factor`
    - `rotate_factor`
    - `random_flip_prob`
- Update `UserConfig` to include an `augments` field of type `AugmentationSection`.
- Ensure proper serialization and deserialization support for the new `[augments]` TOML section.

### 2.2. API Enhancement (`EZMMLab.train`)
- Update the `train` method signature in `EZMMLab` (and concrete subclasses like `RTMDet`, `RTMPose`, `RTMO`) to accept a single `augments: Optional[Dict[str, Any]] = None` argument.
- Implement **Strict Validation** in `train()`:
    - Validate that keys provided in the `augments` dictionary are supported by the `PipelineTransformPatcherRegistry` for the specific model family.
    - Raise a `ValueError` with a descriptive message if an unsupported key is provided.
- Implement **Dynamic Docstrings** for the `train` methods to list the currently available augmentations for that specific model at runtime.

### 2.3. Configuration Management
- Update `ConfigManager.create_fresh_config` to handle the transition of augmentation parameters into the new `AugmentationSection`.
- Ensure the `augments` dictionary passed to `train()` is correctly mapped to the `AugmentationSection` in the resulting `UserConfig`.

### 2.4. Injector Refactoring ("Clean Up")
- Refactor `MMDetInjector._patch_pipelines` and `MMPoseInjector._patch_pipelines` to:
    - Consume augmentation parameters exclusively from the new `user_config.augments` section.
    - Ensure clean, decoupled logic for transform patching and dynamic insertion (e.g., `RandomFlip`).
    - Remove redundant or misplaced augmentation logic from other parts of the injectors to ensure a single source of truth.

## 3. Non-Functional Requirements
- **Data Integrity:** Leverage Pydantic's validation to enforce types and constraints on augmentation values.
- **SOLID Compliance:** Maintain Single Responsibility by keeping augmentation definitions decoupled from training orchestration.
- **Scalability:** The registry-based approach ensures that adding new augmentations only requires adding a new patcher and registering it.

## 4. Acceptance Criteria
- [x] `UserConfig` supports a dedicated `[augments]` section in TOML files.
- [x] `model.train(..., augments={"scale_factor": 0.5})` successfully applies the augmentation.
- [x] Providing an unsupported augmentation key (e.g., `augments={"invalid_aug": 1.0}`) raises a `ValueError` specifying the available keys for that model family.
- [x] The `train()` method docstring dynamically lists available augmentations for the model (e.g., RTMDet vs RTMPose).
- [x] Injectors are refactored to be cleaner and use the consolidated `augments` structure.
- [x] Existing unit and integration tests pass with the refactored structure.
- [x] New unit tests validate the `AugmentationSection` and the `train()` validation logic.

## 5. Out of Scope
- Implementation of entirely new augmentation transforms (this track focuses on reorganization and API).
- Modification of inference-time augmentations (TTA is already handled by specific patchers).
