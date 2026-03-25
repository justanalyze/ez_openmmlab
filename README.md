# 🚀 ez_openmmlab: OpenMMLab Made EZ

[![PyPI version](https://badge.fury.io/py/ez-openmmlab.svg)](https://pypi.org/project/ez-openmmlab/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> OpenMMLab has some of the best CV models available — RTMDet, RTMPose, RTMO.
> But actually using them is brutal. ez_openmmlab fixes that.

![Training Demo](docs/train.gif)

---

## See It In Action

**Training** — from zero to running in 4 lines:
```python
from ez_openmmlab import RTMDet

model = RTMDet("rtmdet_tiny")
model.train(dataset_config_path="dataset.toml", epochs=100)
```

**Inference:**
```python
results = model.predict("image.jpg", show=True)

for box in results[0].boxes:
    print(f"Class: {box.cls}, Score: {box.conf:.3f}")
```

**Export to ONNX or TensorRT:**
```python
model.export(format="onnx", image="image.jpg", output_dir="deploy/")
```

---

## Why ez_openmmlab?

| Feature | Traditional OpenMMLab | The EZ Way |
| :--- | :--- | :--- |
| **Setup** | Hours of dependency archaeology | Under 5 minutes with `uv` |
| **Config** | Inheriting through 5+ Python files | One human-readable `.toml` |
| **Data** | Fighting with Dataset Registries | Just point to your dataset |
| **Results** | Complex dictionary structures | Vectorized, NumPy-first objects |
| **Deploy** | Hours installing MMDeploy | Just call `.export()` |

---

## Installation

ez_openmmlab uses **uv** for fast, reliable installation.

**1. Install uv:**
```bash
# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Create virtual environment:**
```bash
uv venv -p 3.10
source .venv/bin/activate  # Linux
# .venv\Scripts\activate   # Windows
```

**3. Install (GPU — CUDA 11.7):**
```bash
uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu117
uv pip install mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
uv pip install git+https://github.com/JustAnalyze/chumpy.git@master
uv pip install ez-openmmlab
```

**3. Install (CPU):**
```bash
uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
uv pip install git+https://github.com/JustAnalyze/chumpy.git@master
uv pip install ez-openmmlab
```

> Don't want to use uv? See [install/README.md](install/README.md)

---

## Full Usage

### 🏋️ Training

Define your data in a simple `dataset.toml`:
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

> [!IMPORTANT]
> The `classes` list must exactly match the `categories` in your COCO 
> annotation files — same names, same order.
```python
from ez_openmmlab import RTMDet

model = RTMDet("rtmdet_tiny")
model.train(
    dataset_config_path="dataset.toml",
    epochs=100,
    batch_size=16,
)
```

> [!TIP]
> Training interrupted? Resume with:
> `model = RTMDet(model="user_config.toml")` then `model.resume()`

> [!IMPORTANT]
> **Windows users:** Always wrap training in 
> `if __name__ == "__main__":` to avoid multiprocessing errors.

![Training Demo](docs/train.gif)

---

### 🔍 Inference

![Inference Demo](docs/inference.gif)
```python
from ez_openmmlab import RTMDet

# Your trained model
model = RTMDet(model="user_config.toml", checkpoint_path="epoch_100.pth")

# Or a pretrained model
model = RTMDet("rtmdet_s")

results = model.predict("image.jpg", show=True)

for box in results[0].boxes:
    print(f"Class: {box.cls}, Score: {box.conf:.3f}, BBox: {box.xyxy}")
```

---

### 🚢 Export

![Export Demo](docs/export.gif)

> [!IMPORTANT]
> **Docker is required** for `.export()`. 
> The MMDeploy image is large (30GB+) and will be downloaded on first use.
```python
from ez_openmmlab import RTMDet

model = RTMDet(model="user_config.toml", checkpoint_path="epoch_100.pth")

model.export(
    format="onnx",        # or 'tensorrt'
    image="sample.jpg",
    output_dir="deploy/",
    device="cpu"          # use 'cuda' for TensorRT
)
```

---

### 🧘 Custom Pose Estimation
```python
from ez_openmmlab import RTMPose

model = RTMPose("rtmpose_s")
model.train(dataset_config_path="pose_dataset.toml", epochs=210)

results = model.predict("person.jpg", show=True)
for person in results[0].keypoints:
    print(f"Keypoints: {person.xy}")
```

See the [Full Pose Configuration Guide](docs/pose_guide.md)

---

## ✨ Key Features

- **EZ Environment** — Reproducible setups via `uv`
- **EZ Configuration** — Human-readable TOML replaces config inheritance
- **Auto Checkpoints** — Missing weights downloaded automatically
- **Strict Validation** — Pydantic catches errors before your run starts
- **Clean Results** — Vectorized, NumPy-first output objects
- **Flexible Loading** — Pretrained or custom checkpoints, same API

---

## 🗺️ Roadmap

- [x] Resume training from interrupted sessions
- [x] One-click `.export()` to ONNX and TensorRT
- [ ] Full CLI — train and infer from terminal
- [ ] Architecture expansion — more OpenMMLab models

**Currently Supported:**
- Detection & Segmentation: `rtmdet`, `rtmdet-ins`
- 2D Pose Estimation: `rtmpose`, `rtmo`

---

## 🤝 Acknowledgements

`ez_openmmlab` wouldn't exist without the research and engineering of 
the **OpenMMLab** team.

---

## 📄 License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

## 🐛 Issues & Contributions

Found a bug? Have a feature request? 
[Open an issue](https://github.com/JustAnalyze/ez_openmmlab/issues)

Pull requests are welcome. For major changes please open an issue first.

---

## 📖 Resources

- [Demo Examples](demos/) — Complete end-to-end workflows
- [Issues](https://github.com/JustAnalyze/ez_openmmlab/issues) — 
  Bugs and feature requests
