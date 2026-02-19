# Specification: Dynamic Session Dataset Registration

## Overview
This track implements a dynamic registration system for custom datasets. Instead of passing `metainfo` as a dictionary to a generic dataset class, `ez_openmmlab` will dynamically create a new Python class at runtime, hardcode its `METAINFO`, and register it with the OpenMMLab `DATASETS` registry. This ensures that evaluators, visualizers, and data pipelines correctly recognize the custom dataset structure.

## Functional Requirements
- **Dynamic Class Creation:** Use the Python `type()` function to create classes that inherit from `BaseCocoStyleDataset` (MMPose) or `CocoDataset` (MMDet) during the `train()` execution.
- **Naming Convention:** The new class will be named based on the `dataset_name` found in the configuration (e.g., `MyCustomData` -> `MyCustomDataDataset`).
- **Registry Integration:**
    - Use `mmpose.registry.DATASETS.register_module()` for pose models.
    - Use `mmdet.registry.DATASETS.register_module()` for detection models.
- **Config Rewriting:** Automatically update the `train_dataloader` and `val_dataloader` configurations to use the newly registered class name as the `type` parameter.
- **Collision Handling:** Throw a `ValueError` if a class with the same name is already registered, requiring the user to specify a unique `dataset_name`.

## Non-Functional Requirements
- **Session Scoped:** Registration only needs to persist for the duration of the Python process.
- **Transparency:** The user should not need to write any Python code to define the dataset class; it should be handled entirely via the `dataset.toml`.

## Acceptance Criteria
- [ ] Training on a custom dataset uses a uniquely named registered class instead of a generic base class.
- [ ] The `metainfo` (sigmas, skeleton, classes) is correctly picked up by the `CocoMetric` evaluator without explicit passing.
- [ ] Providing a duplicate `dataset_name` that is already in the registry triggers a clear error message.
- [ ] Both MMDet and MMPose engines support this dynamic registration.

## Out of Scope
- Support for non-COCO style datasets (e.g., VOC, XML) in this phase.
- Persistent disk-based class generation (everything happens in memory).
