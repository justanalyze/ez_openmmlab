# Implementation Plan: Support Custom Model Loading via config.toml

## Phase 1: Core Base Enhancements

- [x] Task: Update `EZMMLab` constructor signature [f8bd4dd]
    - [x] Modify `__init__` in `src/ez_openmmlab/core/base.py` to accept `model: Union[str, Path, ModelName]`.
    - [x] Update type hints and docstrings.
- [x] Task: Implement Custom Config Loading Logic [f8bd4dd]
    - [x] Create private method `_resolve_model_config(self, model_input: Union[str, Path, ModelName])`.
    - [x] Implement logic:
        - [x] If input is `ModelName` (or string equivalent): Use existing default behavior (fetch base weights/config).
        - [x] If input is path to `.toml`:
            - [x] Validate existence of `checkpoint_path`. Raise `ValueError` if missing.
            - [x] Parse `config.toml` to get `model.name` and other settings.
            - [x] Resolve base config path using `ModelName(model_name).config_path`.
            - [x] Create a temporary `.py` config file that inherits from the base config and applies overrides from `.toml`.
            - [x] Return the path to this temporary `.py` file as the config to be used.
- [x] Task: Cleanup Mechanism [f8bd4dd]
    - [x] Ensure temporary `.py` files are deleted after initialization (or use `tempfile` module contexts).
- [x] Task: Conductor - User Manual Verification 'Core Base Enhancements' (Protocol in workflow.md) [f8bd4dd]

## Phase 2: Engine Updates

- [x] Task: Update `EZMMDetector` [4c805eb]
    - [x] Ensure `EZMMDetector` inherits the new constructor logic.
    - [x] Verify `_init_inferencer` uses the resolved config path.
- [x] Task: Update `EZMMPose` [e25cfd7]
    - [ ] Ensure `EZMMPose` (and subclasses `RTMPose`, `RTMO`) inherits the new constructor logic.
    - [ ] Verify `_init_inferencer` uses the resolved config path.
- [ ] Task: Conductor - User Manual Verification 'Engine Updates' (Protocol in workflow.md)

## Phase 3: Artifact Management

- [ ] Task: Update Config Saving
    - [ ] Modify `save_user_config` (or call site in `base.py`) to include `model.base_config_path` in the saved `user_config.toml`.
    - [ ] Ensure this path resolves to the absolute path of the generated `.py` config file created in the workdir during training.
- [ ] Task: Conductor - User Manual Verification 'Artifact Management' (Protocol in workflow.md)

## Phase 4: Verification

- [ ] Task: Unit Tests
    - [ ] Test `EZMMLab` constructor with valid `ModelName`.
    - [ ] Test `EZMMLab` constructor with `config.toml` + `checkpoint`.
    - [ ] Test `EZMMLab` constructor with `config.toml` missing `checkpoint` (expect failure).
- [ ] Task: Integration Test
    - [ ] Simulate a full flow: Train a model -> Save `config.toml` -> Load new model instance using that `config.toml` + checkpoint -> Run inference.
- [ ] Task: Conductor - User Manual Verification 'Verification' (Protocol in workflow.md)
