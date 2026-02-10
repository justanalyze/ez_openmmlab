# 🚀 ez_openmmlab: Openmmlab Made EZ

`ez_openmmlab` is a streamlined Python wrapper designed to unlock the full potential of the [OpenMMLab](https://github.com/open-mmlab) ecosystem. It eliminates the traditional "OpenMMLab Tax", the complexity of nested Python configurations and notorious dependency conflicts, providing a unified, "Config-First" workflow.

---

## 💡 Why ez_openmmlab?

OpenMMLab produces State-of-the-Art (SOTA) models like **RTMDet**, **RTMPose**, and **RTMO** that consistently dominate benchmarks. However, for many developers, the barrier to entry is high:

1.  **The Dependency Whack-a-Mole:** Installing MMDet or MMPose often feels like a frustrating game of whack-a-mole, trying to align specific versions of Torch, MMCV, and CUDA. `ez_openmmlab` resolves this "dependency hell" with optimized, reproducible environments.
2.  **Config Fatigue:** Customizing a model for your own dataset usually requires editing 500-line Python files spread across multiple directories. We've replaced this with a simple, human-readable **TOML-First** workflow.
3.  **High-Bar Complexity:** Taking advantage of these SOTA models shouldn't require deep expertise in framework internals. We provide the power of OpenMMLab without the stress.

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
# Clone the repository (including the submodules)
git clone --recursive https://github.com/JustAnalyze/ez_openmmlab.git
cd ez_openmmlab

# Install for CPU
uv sync --extra cpu

# OR: Install for GPU (CUDA 11.7)
uv sync --extra gpu
```

---

## 📖 Quick Start

### 1. Define your Data (`dataset.toml`)

Describe your dataset once. Forget about framework internals.

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

### 2. Train SOTA Models

```python
from ez_openmmlab import RTMDet

# Initialize (choices: rtmdet_tiny, rtmdet_s, rtmdet-ins_m, etc.)
detector = RTMDet("rtmdet_tiny")

# Start training on your custom data
detector.train(
    dataset_config_path="dataset.toml",
    epochs=100,
    batch_size=16
)
```

### 3. Inference Made Simple

```python
from ez_openmmlab import RTMPose

# Load a model with custom weights easily
model = RTMPose(
    model="rtmpose_s",
)

results = model.predict("sample.jpg", show=True)

for bbox in results[0].boxes:
    print(f"Bbox: {bbox.xyxy}, Class {bbox.cls}, Confidence: {bbox.conf}")
```

---

## 🗺️ Roadmap

- [ ] **Native Export:** One-click `.export()` to ONNX and TensorRT for production. (For now install [MMDeploy](https://github.com/open-mmlab/mmdeploy) Via Docker to prevent headaches ;) )
- [ ] **Architecture Expansion:** Bringing the "EZ" treatment to more SOTA architectures under OpenMMLab (taking suggestions!).
- [ ] **Full CLI:** Run entire training and inference experiments directly from your terminal.

---

## 🤝 Acknowledgements

`ez_openmmlab` wouldn't exist without the relentless research and engineering of the **OpenMMLab** team. Their models are world-class and still competing with (and beating) the newest architectures out there.

We aren't replacing OpenMMLab; we're **amplifying** it. We want to make sure every developer can take advantage of these incredible tools without being held back by the complexity of the stack.

**Currently Supported:**

- **Detection & Segmentation:** `rtmdet` (all variants)
- **2D Pose Estimation:** `rtmpose`
- **Multi-Person Pose:** `rtmo`
