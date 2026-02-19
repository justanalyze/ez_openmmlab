# Specification: Terminology Alignment - Rename Handlers to Injectors

## Overview
Rename the `handlers` module to `injectors` to more accurately reflect their purpose: injecting user-defined configurations into the base OpenMMLab `Config` objects. This change aligns the codebase with the "Configuration Injection" mental model and improves architectural clarity.

## Goals
- Rename the physical directory and files to use `injectors`.
- Update all class names, method names, and variable names to use `injector` terminology.
- Ensure 100% backward compatibility within the internal API during the transition (no public API changes are expected as these are internal core components).

## Functional Requirements
1.  **Refactor Directory Structure:**
    - Rename `src/ez_openmmlab/core/handlers/` to `src/ez_openmmlab/core/injectors/`.
2.  **Rename Classes and Interfaces:**
    - `BaseConfigHandler` -> `BaseConfigInjector`
    - `DataloaderHandler` -> `DataloaderInjector`
    - `RuntimeHandler` -> `RuntimeInjector`
    - `MMDetHandler` -> `MMDetInjector`
    - `MMPoseHandler` -> `MMPoseInjector`
3.  **Rename Registry and Logic:**
    - `get_handlers()` -> `get_injectors()`
    - Update internal loops and variable names (e.g., `for handler in handlers` -> `for injector in injectors`).
4.  **Update Imports:**
    - Update all import statements in `src/ez_openmmlab/core/base.py` and other modules referencing the old `handlers` path.

## Non-Functional Requirements
- **Maintainability:** Terminological consistency makes the codebase easier for new contributors to understand.
- **Stability:** Existing training and inference workflows must remain functional.

## Acceptance Criteria
- No files or directories named `handlers` exist in `src/ez_openmmlab/core/`.
- The string `handler` (case-insensitive) is removed from all configuration patching logic.
- `uv run pytest` passes all unit and integration tests.
- Training and inference demos continue to work without modification.

## Out of Scope
- Refactoring the actual logic within the injectors.
- Renaming unrelated components (e.g., event handlers in MMEngine if any).
