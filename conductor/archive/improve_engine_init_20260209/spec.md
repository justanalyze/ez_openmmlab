# Specification: Enhance EZMMLab Initialization Readability

## Overview
Refactor the `__init__` method of the `EZMMLab` base class in `src/ez_openmmlab/core/engines/engine_base.py` to improve readability, maintainability, and adherence to SOLID principles. The goal is to transform a dense initialization block into a clear, self-documenting sequence of logical steps.

## Goals
- **De-clutter `__init__`:** Transform the constructor into a high-level "recipe" of initialization steps.
- **Improved Cohesion:** Group related initialization logic into dedicated private methods.
- **Explicit State:** Clearly group and document class attributes.
- **Robust Validation:** Extract resource validation into a distinct, early step.

## Functional Requirements
1.  **Attribute Organization:**
    - Explicitly group attributes into sections: `Logging & Internal State`, `Model Configuration`, and `Internal Managers`.
2.  **Logic Extraction:**
    - Extract initialization logic into the following private methods:
        - `_configure_logging(log_level)`
        - `_validate_inputs(model, checkpoint_path)`: Handle the `.toml` and checkpoint consistency check.
        - `_resolve_resources(model, checkpoint_path)`: Resolve the actual checkpoint and config paths.
        - `_initialize_model_state(model)`: Set the `model` name and auto-load metadata.
3.  **Consistency:**
    - Ensure that the refactor does not change any external behavior or side-effects (e.g., noisy warning suppression, logging configuration).

## Non-Functional Requirements
- **Readability:** The code should be easy for a new developer to follow at a glance.
- **SOLID Compliance:** Adherence to the Single Responsibility Principle by separating setup concerns.

## Acceptance Criteria
- [ ] `EZMMLab.__init__` is reduced to a concise sequence of method calls.
- [ ] All new private methods are well-documented with docstrings.
- [ ] No regressions in existing unit or integration tests.
- [ ] Logging behavior remains consistent with the current implementation.

## Out of Scope
- Refactoring `train()` or `predict()` methods (already covered in previous tracks).
- Modifying `ConfigManager` or other external utilities.
