# Model Integration Guide

This guide outlines the standard procedure for integrating a new OpenMMLab model into the `ez_openmmlab` framework, ensuring it adheres to the project's **SOLID** architecture and **Config-First** philosophy.

---

## Phase 1: Foundation (Schemas & Constants)

Before writing the model logic, you must register its identity and configuration mapping.

1.  **Update `src/ez_openmmlab/schemas/model.py`**:
    *   Add a new entry to the `ModelName` Enum.
    *   Create a dictionary mapping the Enum name to its official OpenMMLab `.py` config path (relative to the library's `configs/` root).
    *   Add the new mapping to the `SUPPORTED_CONFIGS` aggregate.

---

## Phase 2: Parameter Intelligence (Resolvers)

If your model requires derived parameters (e.g., calculating feature map size from input resolution), implement a Resolver.

1.  **Create a Resolver in `src/ez_openmmlab/core/resolvers/`**:
    *   Inherit from `BaseModelParamsResolver`.
    *   Implement `resolve(**kwargs)`: This method should handle defaults and validate parameter compatibility (e.g., ensuring resolution is 32-divisible).
2.  **Register in `ModelParamsResolverFactory`**:
    *   Update `src/ez_openmmlab/core/resolvers/factory.py` to return your new resolver based on the model name.

---

## Phase 3: The Model Engine

This is the primary user-facing class. Place it in `src/ez_openmmlab/models/<library_family>/`.

1.  **Inherit from the Family Base**:
    *   For Detection: `EZMMDetector`
    *   For Pose: `EZMMPose`
    *   For new families: `EZMMLab`
2.  **Implement Required Methods**:
    *   `__init__`: Call `_validate_model(model)` before `super().__init__`.
    *   `_validate_model`: Ensure the model variant is recognized or is a valid `.toml` path.
    *   `_get_architecture_params`: **Critical for Persistence.** Return a dict of parameters (like `input_size`) that should be saved to `config.toml` and reused during inference.
    *   `_get_library_family`: Return the string identifier for the library (e.g., `"mmdet"`, `"mmpose"`). This is used for dynamic dataset registration.
    *   `train`: Explicitly define hyperparameters in the signature (e.g., `input_size`, `weight_decay`) to provide a clean IDE experience, then call `super().train`.
3.  **Specialized Inference (Optional)**:
    *   If the model is Top-Down (like `RTMPose`), override `predict` to handle detector overrides.
    *   Implement `_instantiate_inferencer` to return the correct OpenMMLab Inferencer.

---

## Phase 4: Configuration Patching (Injectors)

If the model requires unique changes to the OpenMMLab `Config` object that aren't covered by global injectors.

1.  **Check Global Injectors**: Ensure `OptimizerInjector` and `EvaluatorInjector` aren't already handling your needs.
2.  **Specialized Patcher (Optional)**:
    *   If you need to patch unique pipeline transforms, add a new Patcher to `src/ez_openmmlab/core/injectors/pipeline_patchers.py` and register it in the `PipelineTransformPatcherRegistry`.
3.  **Update Family Injector**:
    *   Modify `MMDetInjector` or `MMPoseInjector` in `src/ez_openmmlab/core/injectors/` if you need to patch specific model head attributes (like `out_channels`).

---

## Phase 5: Public API Exposure

1.  **Library Level**: Export your class in `src/ez_openmmlab/models/<family>/__init__.py`.
2.  **Top Level**: Export your class in `src/ez_openmmlab/__init__.py` to allow users to import it via `from ez_openmmlab import MyNewModel`.

---

## Phase 6: Verification (Testing)

Every new model must be verified with both unit and integration tests.

1.  **Unit Tests**: Create a new test file in `tests/unit/` (e.g., `test_my_model.py`).
    *   Verify that `_get_architecture_params` returns expected values.
    *   Mock the Inferencer to ensure `predict` calls it with patched configurations.
2.  **Integration Tests**: Add a smoke test in `tests/integration/test_smoke.py` or a dedicated integration test to verify the full inference loop with a sample image.

---

## Implementation Checklist

| Component | Responsibility | File Location |
| :--- | :--- | :--- |
| **Enum** | Identity & Config Path | `schemas/model.py` |
| **Resolver** | Logic for derived params | `core/resolvers/` |
| **Injector** | Patching the `.py` Config | `core/injectors/` |
| **Patcher** | Specific Pipeline Transforms | `core/injectors/pipeline_patchers.py` |
| **Model Class** | User Interface (Train/Predict) | `models/<family>/<name>.py` |

---

## Pro-Tip: The "Golden Rule" of `_get_architecture_params`

Always ensure every parameter you want saved into the `metadata` of a trained model is returned by `_get_architecture_params`. This ensures that when a user runs `RTMPose(model="path/to/config.toml")`, the engine automatically knows the `input_size` and `simcc_sigma` without the user providing them again.