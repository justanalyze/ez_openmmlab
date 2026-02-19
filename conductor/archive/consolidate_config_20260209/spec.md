# Specification: Consolidate and Refine Configuration Management

## Overview
This refactoring track focuses on consolidating and refining the configuration management logic within the `core` module. We will merge the existing `config_loader.py` and `config_builder.py` into a single, cohesive `config_manager.py`, rename the primary utility class to `ConfigManager`, and address several TODOs to improve code quality, performance, and clarity.

## Goals
- **Consolidate Logic:** Merge `BaseConfigLoader` and `UserConfigBuilder` into `ConfigManager`.
- **Improve Performance:** Narrow the metadata search scope to the immediate parent of the checkpoint.
- **Enhance Readability:** Use explicit module namespaces for utility calls and rename methods for clarity.
- **Maintain SOLID:** Ensure the new `ConfigManager` maintains a clear responsibility for all configuration-related orchestration.

## Functional Requirements
1.  **Module Consolidation:**
    - Create `src/ez_openmmlab/core/config_manager.py`.
    - Move `BaseConfigLoader` and the global `get_config_file` utility into this new file.
    - Move `UserConfigBuilder` into this new file and rename it to `ConfigManager`.
2.  **API Refinement:**
    - Rename `ConfigManager.build()` to `ConfigManager.build_user_config()`.
    - Update all internal and external references to these renamed components.
3.  **Metadata Search Optimization:**
    - Update `load_metadata_from_checkpoint()` to *only* search the immediate parent directory (`checkpoint_path.parent`).
    - Restrict the search to only look for `user_config.toml`. Stop searching for `dataset.toml`.
4.  **Codestyle Alignment:**
    - Update imports to use `from ez_openmmlab.utils import toml_config`.
    - Replace direct calls like `load_user_config()` with module-qualified calls: `toml_config.load_user_config()`.

## Non-Functional Requirements
- **No Functional Regression:** Existing training and inference workflows must continue to work seamlessly.
- **Clean Architecture:** The consolidated file should be logically organized, with clear separation between base config resolution and user config assembly.

## Acceptance Criteria
- [ ] `src/ez_openmmlab/core/config_loader.py` and `src/ez_openmmlab/core/config_builder.py` are deleted.
- [ ] A single `src/ez_openmmlab/core/config_manager.py` handles all config logic.
- [ ] Metadata is only auto-loaded from `user_config.toml` in the immediate checkpoint parent.
- [ ] All `toml_config` utility calls are module-qualified.
- [ ] `uv run pytest` passes for all unit and integration tests.

## Out of Scope
- Changing the schema of `user_config.toml` or `dataset.toml`.
- Modifying the underlying MMLab configuration loading mechanisms (MMEngine).
