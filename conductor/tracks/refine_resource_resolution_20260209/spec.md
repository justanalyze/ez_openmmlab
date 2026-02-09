# Specification: Refine Resource Resolution and Enforce Explicit Configuration

## Overview
Simplify and improve the readability of the resource resolution logic in `EZMMLab`. This refactor eliminates implicit filesystem inference of model metadata, favoring explicit user-provided configurations via `config.toml`. The `_initialize_model_state` method will be removed, and its responsibilities will be integrated into a cleaner, branch-based `_resolve_resources` method.

## Goals
- **Explicit Metadata Loading**: Stop searching for metadata files near the checkpoint. Rely solely on the provided `model` input (whether a `ModelName` or a path to `config.toml`).
- **Simplify `_resolve_resources`**: Use clear branching to separate custom `.toml` loading from standard `ModelName` resolution.
- **Stateless/Clean API**: Update `ConfigManager` to return metadata dicts directly from TOML.
- **De-clutter Base Engine**: Remove `_initialize_model_state`.

## Functional Requirements
1.  **Update `ConfigManager`**:
    - Implement `load_metadata_from_toml(path: Path) -> Dict[str, Any]` to extract `model_name`, `num_classes`, `num_keypoints`, and `metainfo` from a `user_config.toml`.
2.  **Refactor `_resolve_resources` in `EZMMLab`**:
    - Implement a clear `if/else` structure based on whether the `model` input is a `.toml` file.
    - **Custom Path**:
        - Set `self.checkpoint_path` directly from the user argument.
        - Call the new `ConfigManager` metadata loader to populate all model state attributes (`self.num_classes`, etc.).
        - Generate the temporary Python config.
    - **Standard Path**:
        - Resolve resources using `ensure_model_checkpoint` and `get_config_file`.
        - Set `self.model` name.
3.  **Remove Legacy Logic**:
    - Delete `_initialize_model_state` from `EZMMLab`.
    - Delete the broad filesystem search logic in `ConfigManager` that looked for metadata in parent directories.

## Non-Functional Requirements
- **Explicit over Implicit**: The system must not "guess" configuration values from the environment.
- **Readability**: The initialization sequence should be a flat, easy-to-read block of logic.

## Acceptance Criteria
- [ ] `EZMMLab` has no `_initialize_model_state` method.
- [ ] `_resolve_resources` is the single source of truth for asset resolution.
- [ ] Custom model loading strictly uses the values provided in the `.toml`.
- [ ] All unit tests for custom and standard initialization pass.

## Out of Scope
- Modifying the public signatures of `train()` or `predict()`.
- Refactoring the result formatters.
