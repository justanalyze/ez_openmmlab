# 🚀 ez_openmmlab: OpenMMLab Made EZ
[![PyPI version](https://badge.fury.io/py/ez-openmmlab.svg)](https://pypi.org/project/ez-openmmlab/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Utilize OpenMMLab using an EZ and Familiar API ;)

---

## 🧐 Why ez_openmmlab?

| Feature | The Traditional Way (OpenMMLab) | The **EZ** Way |
| :--- | :--- | :--- |
| **Setup** | Hours of dependency archaeology | less than 5 minutes with `uv` |
| **Config** | Inheriting through 5+ Python files | **One** human-readable `.toml` |
| **Data** | Fighting with Dataset Registries | Just point to your datasets `dataset.toml`|
| **Results** | Complex dictionary structures | Vectorized, NumPy-first objects |
| **Deploy** | Spend Hours installing and learning MMDeploy | just call `.export()` method |

---
> 💡 **New to ez_openmmlab?** Check out the [`demos/`](demos/) folder for complete end-to-end examples!
---

## 🏋️ 1. Train

Forget framework-level "surgery". Define your data in a simple `dataset.toml`, call `.train()`, and `ez_openmmlab` handles the rest.

### Step A: Define your data (`dataset.toml`)

No more manual registration. Just point to your files.

> [!IMPORTANT] The `classes` list must exactly match the `categories` in your COCO annotation files. For example, if your `train.json` contains:
>
> ```json
>
> "categories": [
>   {"id": 1, "name": "cat"},
>   {"id": 2, "name": "dog"}
> ]
>
> ```
>
> Then your `dataset.toml` should have `classes = ["cat", "dog"]` in the same order.

```toml
dataset_name = "MY_CUSTOM_DATASET"
data_root = "datasets/my_project"
classes = ["cat", "dog"]

[train]
ann_file = "annotations/train.json"
img_dir = "images/train"

[val]
ann_file = "annotations/val.json"
img_dir = "images/val"
```

### Step B: Launch Training

One method. I'm sure this is familiar for most of you ;)

```python
from ez_openmmlab import RTMDet

# Initialize (choices: rtmdet_tiny, rtmdet_s, rtmdet_m, rtmdet_l, rtmdet_x)
model = RTMDet("rtmdet_tiny")

# Start training - outputs user_config.toml for easy reloading
model.train(
    dataset_config_path="dataset.toml",
    epochs=100,
    batch_size=16,
)
```

> [!TIP]
> Training got interrupted? Just load your config: `model = RTMDet(model="user_config.toml")` and call `model.resume()`

![Training Demo](docs/train.gif)

---

## 🔍 2. Inference

Load your trained model or use pretrained weights. Predict and visualize with a single line.

```python
from ez_openmmlab import RTMDet

# Option 1: Load your trained model
model = RTMDet(
    model="user_config.toml",      # Config generated during training
    checkpoint_path="epoch_100.pth"  # Provide which checkpoint you want to load
)

# Option 2: Or just use a pretrained model for quick inference
model = RTMDet("rtmdet_s")

# Inference made simple
results = model.predict("sample.jpg", show=True)

# Access clean, structured results
for box in results[0].boxes:
    print(f"Class: {box.cls}, Score: {box.conf:.3f}, BBox: {box.xyxy}")
```

![Inference Demo](docs/inference.gif)

---

## 🚢 3. Export

Deploying to production with mmdeploy is usually a nightmare. We simplified it to one command using **MMDeploy via Docker**.

> [!IMPORTANT]
> **Docker is required** for the `.export()` method. If the MMDeploy image is missing, you will be prompted to download it (warning: 30GB+).

```python
from ez_openmmlab import RTMDet

# Load your model (trained or pretrained)
model = RTMDet(
    model="user_config.toml",
    checkpoint_path="epoch_100.pth"
)

# Export to ONNX or TensorRT
model.export(
    format="onnx",        # Options: 'onnx' or 'tensorrt'
    image="sample.jpg",   # Required for model tracing
    output_dir="deploy/", # Where to save artifacts
    device="cpu"          # Use 'cuda' for TensorRT
)
```

![Export Demo](docs/export.gif)

---

## 🧘 Custom Pose Estimation? Still EZ

Training on custom keypoints? Just add your metainfo to the TOML. **You can add as many keypoints as your dataset requires.**

> [!IMPORTANT]
> **Keypoints and skeleton must match your COCO annotations!**
> 
> - `keypoint_info` must match the `keypoints` list in your COCO JSON
> - `skeleton_info` must match the `skeleton` connections in your COCO JSON
> - Order and names must be identical

**Example COCO JSON structure:**

```json
"keypoints": ["nose", "left_eye", "right_eye"],
"skeleton": [[0, 1], [0, 2]]
```

**Corresponding dataset.toml:**

```toml
# pose_dataset.toml
dataset_name = "MY_CUSTOM_POSE_DATASET"
data_root = "datasets/custom_pose"
classes = ["person"]

[train]
ann_file = "annotations/train.json"
img_dir = "images/train"

[val]
ann_file = "annotations/val.json"
img_dir = "images/val"

[metainfo]
sigmas = [0.025, 0.025, 0.05]        # One per keypoint
joint_weights = [1.0, 1.0, 1.0]      # One per keypoint

[metainfo.keypoint_info.0]
name = "nose"                         # Must match COCO keypoints[0]
id = 0
color = [51, 153, 255]

[metainfo.keypoint_info.1]
name = "left_eye"                     # Must match COCO keypoints[1]
id = 1
color = [51, 153, 255]

[metainfo.keypoint_info.2]
name = "right_eye"                    # Must match COCO keypoints[2]
id = 2
color = [51, 153, 255]

[metainfo.skeleton_info.0]
link = ["nose", "left_eye"]           # Must match COCO skeleton[0]
id = 0

[metainfo.skeleton_info.1]
link = ["nose", "right_eye"]          # Must match COCO skeleton[1]
id = 1

# Add as many keypoints and skeleton connections as needed
```

```python
from ez_openmmlab import RTMPose

# Initialize (choices: rtmpose_tiny, rtmpose_s, rtmpose_m, ...)
model = RTMPose("rtmpose_s")
model.train(dataset_config_path="pose_dataset.toml", epochs=210)

# Inference with your custom keypoints
results = model.predict("person.jpg", show=True)
```

---

## 🛠️ Installation

ez-openmmlab uses **uv** for fast, reliable installations (10-100x faster than pip).

### Quick Start

**1. Install uv:**

```bash
# on linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# on windows powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Create virtual environment:**

```bash
uv venv -p 3.10 # or 3.9
source .venv/bin/activate  # Linux
# .venv\Scripts\activate    # Windows
```

**3. Install ez-openmmlab:**

**GPU (CUDA 11.7):**

```bash
# Step 1: Install PyTorch
uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# Step 2: Install MMCV
uv pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html

# Step 3: Install chumpy
# MAKE SURE THAT YOU HAVE GIT INSTALLED.
uv pip install git+https://github.com/JustAnalyze/chumpy.git@master

# Step 4: Install ez-openmmlab
uv pip install ez-openmmlab
```

**CPU:**

```bash
# Step 1: Install PyTorch
uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install MMCV
uv pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html

# Step 3: Install chumpy
# MAKE SURE THAT YOU HAVE GIT INSTALLED.
uv pip install git+https://github.com/JustAnalyze/chumpy.git@master

# Step 4: Install ez-openmmlab
uv pip install ez-openmmlab
```

### Other Installation methods

Don't want to use uv? See [install/README.md](install/README.md) for manual pip installation instructions.

### Requirements

- Python 3.9 or 3.10
- uv package manager
- NVIDIA GPU with CUDA 11.7 (for GPU version)
- Git

## Troubleshooting

### Virtual environment not activated

Make sure you see `(.venv)` at the beginning of your terminal prompt. If not:

```bash
source .venv/bin/activate  # Linux
.venv\Scripts\activate      # Windows
```

### uv not found after installation

Restart your terminal or run:

```bash
source $HOME/.cargo/env
```

## ✨ Key Features

- **EZ Environment:** Reproducible setups that just work via `uv`.
- **EZ Configuration:** Human-readable TOML replaces complex Python config inheritance.
- **Auto-Magic Checkpoints:** Missing weights? We download them for you automatically.
- **Strict Validation:** Powered by Pydantic to catch errors _before_ you start your run.
- **Performance Optimized:** Vectorized, NumPy-first results with **Lazy Initialization**.
- **Flexible Model Loading:** Load pretrained models or your own trained checkpoints seamlessly.

---

## 📚 Quick Start Examples

### Object Detection Workflow

```python
from ez_openmmlab import RTMDet

# 1. Train on custom data
model = RTMDet("rtmdet_s")
model.train(dataset_config_path="dataset.toml", epochs=100)

# 2. Inference with trained model
model = RTMDet(model="user_config.toml", checkpoint_path="epoch_100.pth")
results = model.predict("test_image.jpg", show=True)

# 3. Export for deployment
model.export(format="onnx", image="test_image.jpg", output_dir="deploy/")
```

### Pose Estimation Workflow

```python
from ez_openmmlab import RTMPose

# 1. Train on custom keypoints
model = RTMPose("rtmpose_m")
model.train(dataset_config_path="pose_dataset.toml", epochs=210)

# 2. Inference
model = RTMPose(model="user_config.toml", checkpoint_path="best_model.pth")
results = model.predict("person.jpg", show=True)

# Access keypoint coordinates
for person in results[0].keypoints:
    print(f"Keypoints: {person.xy}")  # Shape: [num_keypoints, 2]
```

---

## 🗺️ Roadmap

- [x] **Resume Training:** Continue from interrupted training sessions.
- [x] **Native Export:** One-click `.export()` to ONNX and TensorRT.
- [ ] **Full CLI:** Run training and inference directly from your terminal.
- [ ] **Architecture Expansion:** Bringing the "EZ" treatment to more OpenMMLab models. (This is a good candidate: https://github.com/53mins/CIGPose)

---

## 🤝 Acknowledgements

`ez_openmmlab` wouldn't exist without the relentless research and engineering of the **OpenMMLab** team.

**Currently Supported:**

- **Detection & Segmentation:** `rtmdet`, `rtmdet-ins`
- **2D Pose Estimation:** `rtmpose`, `rtmo`

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🐛 Issues & Contributions

Found a bug? Have a feature request? [Open an issue](https://github.com/JustAnalyze/ez_openmmlab/issues)!

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📖 Learn More

- **[Demo Examples](demos/)** - Complete end-to-end workflows with datasets
- **[Issues](https://github.com/JustAnalyze/ez_openmmlab/issues)** - Report bugs or request features
