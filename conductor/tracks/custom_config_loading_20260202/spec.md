# Specification: Support Custom Model Loading via config.toml

## Overview
This track aims to simplify the process of loading custom-trained models for inference or further training. Currently, users must manually provide specific parameters. This feature will allow the model constructors (`RTMDet`, `RTMPose`, `RTMO`) to accept either a standard model name (from `ModelName`) or a path to a `config.toml` file generated during a previous training session.

## Functional Requirements
- **Constructor Enhancements:**
    - Update `EZMMLab`, `EZMMDetector`, `EZMMPose`, and their subclasses to accept a `Union[str, Path, ModelName]` for the model parameter.
    - If a valid `ModelName` string/enum is provided, proceed with the existing default loading logic (downloading/fetching base weights).
    - If a path to a `.toml` file is provided, treat it as a custom training configuration.
- **Custom Configuration Logic:**
    - Parse the provided `config.toml`.
    - Infer the base configuration file requirement from the `model.name` stored within the `.toml` (referencing `ModelName.config_path`).
    - Create a temporary OpenMMLab-style `.py` configuration file dynamically based on the base config + `config.toml` settings.
    - Feed this temporary file to the internal engine for initialization.
    - **Automatic Cleanup:** The temporary `.py` file must be deleted immediately after model initialization.
- **Safety & Validation:**
    - If a `config.toml` is provided without an accompanying `checkpoint_path`, the system **must** raise a `ValueError` with an intuitive message explaining that custom configurations require specific checkpoints.
- **Training Artifacts:**
    - Update the `save_user_config` (or equivalent) logic to include the absolute path of the generated `.py` config file (created in the workdir during training) under the `model.base_config_path` key within the saved `config.toml`.

## Non-Functional Requirements
- **Readability:** Maintain a flat and simple structure in the configuration resolution logic.
- **Scalability:** Ensure the implementation in `base.py` is easily inheritable by future engine types.
- **Logging:** Use `loguru` for all warnings and informative messages.

## Acceptance Criteria
- [ ] Model constructors accept `config.toml` paths.
- [ ] Initialization from `config.toml` successfully builds the model head with correct custom parameters (e.g., class counts).
- [ ] Exception is raised if `config.toml` is provided without a checkpoint.
- [ ] `config.toml` saved after training contains the `base_config_path` pointing to the absolute path of the training config.
- [ ] Temporary config files are properly cleaned up.
- [ ] Existing loading via `ModelName` remains fully functional.

## Out of Scope
- Support for loading non-OpenMMLab model formats.
- Modification of existing training loops beyond the metadata saving logic.
