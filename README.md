# 🚀 ez_openmmlab: OpenMMLab Made EZ

Train and deploy SOTA models like **RTMDet**, **RTMPose**, and **RTMO** in minutes, not days. `ez_openmmlab` is a high-level, **TOML-first** wrapper that makes the OpenMMLab ecosystem actually EZ to use.

- **EZ Environment:** Reproducible setups that just work.
- **EZ Configuration:** Human-readable TOML replaces 500-line Python config "surgery".
- **EZ Workflow:** A unified API for training and inference across the entire ecosystem. (MMDet and MMPose so far...)
- **Config Verification:** Validate your final flattened configuration without starting a training run.
- **Unified API:** One interface for Detection, Segmentation, and Pose Estimation.

---

## ✨ Key Features

- **Intuitive API:** Train and predict with simple classes like `RTMDet`, `RTMPose`, and `RTMO`.
- **Config-First Workflow:** Decouple your data from your model using human-readable `dataset.toml` files. No more framework-level "surgery" to change a class count.
- **Auto-Magic Checkpoints:** Missing a model? `ez_openmmlab` automatically downloads official checkpoints to a clean, centralized cache with simplified names.
- **Strict Validation:** Powered by Pydantic to catch configuration and metadata errors (like missing `num_classes` for custom models) _before_ you waste time on a broken run.
- **Performance Optimized:** Features vectorized, NumPy-first results with **Lazy Initialization**—we only process heavy data (like masks) when you actually ask for it.

---

## 🛠️ Installation

`ez_openmmlab` uses [uv](https://github.com/astral-sh/uv) for a lightning-fast and reproducible experience. The provided **Install Script** is the easiest way to handle all complex OpenMMLab dependencies automatically.

```bash
# 1. Clone the repository
git clone https://github.com/JustAnalyze/ez_openmmlab.git
cd ez_openmmlab

# 2. Run the EZ Installer
# This handles uv, submodules, PyTorch, and all build dependencies.
chmod +x install.sh
./install.sh
```

> [!TIP]
> If you just want to try out the library and don't need to export models to production yet, you can **skip** the MMDeploy Docker installation when prompted. The image is **30GB+**. You can always rerun `./install.sh` later to enable export support.

---

## 📖 Quick Start

Choose your task to see how EZ it is:

<details>
<summary><b>🔍 Object Detection & Segmentation (RTMDet)</b></summary>

#### 1. Define Data (`dataset.toml`)

```toml
data_root = "datasets/my_project"
classes = ["cat", "dog"]

[train]
ann_file = "annotations/train.json"
img_dir = "images/train"

[val]
ann_file = "annotations/val.json"
img_dir = "images/val"
```

#### 2. Train & Predict

```python
from ez_openmmlab import RTMDet

# Initialize (choices: rtmdet_tiny, rtmdet_s, rtmdet_m, etc.)
model = RTMDet("rtmdet_tiny")

# Start training on your custom data
model.train(
    dataset_config_path="dataset.toml",
    epochs=100,
    batch_size=16,
    scale_factor=(0.5, 2.0), # Example: Random scaling between 50% and 200%
    random_flip_prob=0.75 # Example: 75% chance of horizontal flip
)

# Inference made simple
results = model.predict("sample.jpg", show=True)
```

</details>

<details>
<summary><b>🧘 Pose Estimation (RTMPose / RTMO)</b></summary>

#### 1. Define Data (`dataset.toml`)

When training pose models on **custom datasets**, you must explicitly define your keypoint identities, skeleton links (for visualization), and sigmas (for OKS evaluation). **You can add as many keypoints as your dataset requires.**

```toml
data_root = "datasets/pose_data"
dataset_name = "my_custom_pose"
classes = ["person"]

[train]
ann_file = "annotations/train.json"
img_dir = "images/train"

[val]
ann_file = "annotations/val.json"
img_dir = "images/val"

[metainfo]
# Sigmas are required for OKS evaluation (one per keypoint)
sigmas = [0.025, 0.025, 0.05]

# Optional: Higher weights (e.g. 2.0) make the model focus more on specific points
joint_weights = [1.0, 1.0, 1.0]

# Define keypoint identities
[metainfo.keypoint_info.0]
name = "nose"
id = 0
color = [51, 153, 255]

[metainfo.keypoint_info.1]
name = "left_eye"
id = 1
color = [0, 255, 0]

# Define skeleton links for visualization
[metainfo.skeleton_info.0]
link = ["nose", "left_eye"]
id = 0
color = [51, 153, 255]
```

#### 2. Train & Predict

```python
from ez_openmmlab import RTMPose

# Initialize (choices: rtmpose_s, rtmpose_m, rtmo_s, etc.)
model = RTMPose("rtmpose_s")

# Start training
model.train(
    dataset_config_path="dataset.toml",
    epochs=210,
    rotate_factor=90.0, # Example: Rotate up to 90 degrees
    random_flip_prob=0.5 # Example: 50% chance of horizontal flip
)

# Inference
results = model.predict("player.jpg", show=True)
```

</details>

---

## 🚀 Model Export (Production)

Deploying OpenMMLab models is often a nightmare due to complex dependencies. `ez_openmmlab` simplifies this by leveraging **MMDeploy via Docker**. Export your models to ONNX or TensorRT with a single command.

### Python API

```python
from ez_openmmlab import RTMDet

model = RTMDet("rtmdet_tiny")
model.export(
    format="onnx",        # Target format: 'onnx' or 'tensorrt'
    image="sample.jpg",   # Required for model tracing
    output_dir="deploy/", # Where to save artifacts
    device="cpu"          # Use 'cuda' for TensorRT
)
```

### CLI

```bash
ez-mmlab export rtmdet_tiny sample.jpg --format onnx --out deploy/
```

---

## 🗺️ Roadmap

- [x] **Native Export:** One-click `.export()` to ONNX and TensorRT for production. (For now install [MMDeploy](https://github.com/open-mmlab/mmdeploy) Via Docker to prevent headaches ;) )
- [ ] **Full CLI:** Run entire training and inference experiments directly from your terminal.
- [ ] **Architecture Expansion:** Bringing the "EZ" treatment to more SOTA architectures under OpenMMLab (taking suggestions!).

---

## 🤝 Acknowledgements

`ez_openmmlab` wouldn't exist without the relentless research and engineering of the **OpenMMLab** team. Their models are world-class and still competing with (and beating) the newest architectures out there.

**Currently Supported:**

- **Detection & Segmentation:** `rtmdet` (all variants)
- **2D Pose Estimation:** `rtmpose`
- **Multi-Person Pose:** `rtmo`
