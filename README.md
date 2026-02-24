# 🚀 ez_openmmlab: OpenMMLab Made EZ

Train and deploy SOTA models like **RTMDet**, **RTMPose**, and **RTMO** in minutes, not days. `ez_openmmlab` is a high-level, **TOML-first** wrapper that makes the OpenMMLab ecosystem actually usable.

- **EZ Environment:** Reproducible setups that just work.
- **EZ Configuration:** Human-readable TOML replaces 500-line Python config "surgery".
- **EZ Workflow:** A unified API for training and inference across the entire ecosystem. (MMDet an MMPose so far...)
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

We recommend using [uv](https://github.com/astral-sh/uv) for a lightning-fast and reproducible experience.

```bash
# 1. Clone the repository (including submodules)
git clone --recursive https://github.com/JustAnalyze/ez_openmmlab.git
cd ez_openmmlab

# 2. Sync the project
uv sync --extra cpu  # For CPU
# OR
uv sync --extra gpu  # For GPU (CUDA 11.7)
```

### 💡 Manual Bootstrap (If `uv sync` fails)

If you encounter issues during installation (common with OpenMMLab's complex build requirements), you can manually bootstrap the environment:

```bash
# 1. Install build dependencies
uv pip install setuptools==80
uv pip install wheel

# 2. Install PyTorch
# For CPU:
uv pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# For GPU (CUDA 11.7):
uv pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117

# 3. Finalize sync
uv sync --extra gpu  # or --extra cpu
```

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

```toml
data_root = "datasets/pose_data"
classes = ["person"] # Usually 'person' for pose estimation

[train]
ann_file = "annotations/person_keypoints_train2017.json"
img_dir = "train2017"

[val]
ann_file = "annotations/person_keypoints_val2017.json"
img_dir = "val2017"
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

## 🗺️ Roadmap

- [ ] **Native Export:** One-click `.export()` to ONNX and TensorRT for production. (For now install [MMDeploy](https://github.com/open-mmlab/mmdeploy) Via Docker to prevent headaches ;) )
- [ ] **Architecture Expansion:** Bringing the "EZ" treatment to more SOTA architectures under OpenMMLab (taking suggestions!).
- [ ] **Full CLI:** Run entire training and inference experiments directly from your terminal.

---

## 🤝 Acknowledgements

`ez_openmmlab` wouldn't exist without the relentless research and engineering of the **OpenMMLab** team. Their models are world-class and still competing with (and beating) the newest architectures out there.

**Currently Supported:**

- **Detection & Segmentation:** `rtmdet` (all variants)
- **2D Pose Estimation:** `rtmpose`
- **Multi-Person Pose:** `rtmo`
