# ez_openmmlab Demos

Welcome to the `ez_openmmlab` demos! This folder contains curated examples showing how to use the library's unified API for detection and pose estimation.

## 🚀 Quick Start

First, ensure you have your virtual environment activated and the project installed:

```bash
# 1. Ensure dependencies are installed
uv sync --extra gpu --preview  # For cuda support (Recommended)
uv sync --extra cpu --preview
```

```bash
# 2. Run any demo using `python` from the **project root**:
.\venv\Scripts\activate # Windows
source .venv/bin/activate  # linux
python demos/demo_rtmdet-bbox.py
```

## 📂 Demo Overview

### 1. Object Detection & Segmentation

- **`demo_rtmdet-bbox.py`**: High-speed object detection. Shows both **single-image** and **batch inference** (passing a list of images).
- **`demo_rtmdet-segmentation.py`**: Instance segmentation (generating masks). Demonstrates batch processing for segmentation tasks.

### 2. Pose Estimation

- **`demo_rtmpose.py`**: **Top-Down** pose estimation. Coordinated pipeline where a detector finds people first, and RTMPose then finds their keypoints.
- **`demo_rtmo.py`**: **Bottom-Up** pose estimation. Fast, one-stage multi-person pose detection. Shows batch processing support.

### 3. Training

- **`demo_train.py`**: Demonstrates the "Config-First" workflow using a simple `dataset.toml` file.

## 🛠️ Key Features Highlighted

- **Batch Inference**: Most `predict()` calls now support passing a `list` of image paths for parallel processing.
- **Unique Output Dirs**: Notice how running a demo twice won't overwrite your results; it automatically increments the output directory (e.g. `runs/demo_x_1`).
- **Pydantic Results**: All predictions are returned as strict, type-safe Python objects.
- **Automatic Weight Management**: Checkpoints are automatically downloaded and cached in the `checkpoints/` folder.
