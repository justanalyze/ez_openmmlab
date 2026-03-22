# 🚀 ez_openmmlab: OpenMMLab Made EZ

Utilize OpenMMLab using an EZ and Familiar API ;)

[![PyPI version](https://badge.fury.io/py/ez-openmmlab.svg)](https://pypi.org/project/ez-openmmlab/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

`ez_openmmlab` is a high-level, **TOML-first** wrapper that makes SOTA OpenMMLab models (**RTMDet**, **RTMPose**, and **RTMO**) actually EZ to use. Stop fighting with 500-line Python configs and dataset registries—just write a few lines of TOML and get back to building.

> 💡 **New to ez_openmmlab?** Check out the [`demos/`](demos/) folder for complete end-to-end examples!

---

## 🏋️ 1. Train

Forget framework-level "surgery". Define your data in a simple `dataset.toml`, call `.train()`, and `ez_openmmlab` handles the rest.

### Step A: Define your data (`dataset.toml`)

No more manual registration. Just point to your files.

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

### Resume Interrupted Training

Training got interrupted? No problem - just resume where you left off:
```python
from ez_openmmlab import RTMDet

# Load from your previous run
model = RTMDet(model="path/to/user_config.toml") # provide the user_config.toml of the interrupted training

# Resume training with new epoch count
model.resume()  # Continues from last checkpoint
```

![Training Demo](docs/train.gif)

---

## 🔍 2. Inference

Load your trained model or use pretrained weights. Predict and visualize with a single line.

```python
from ez_openmmlab import RTMDet

# Option 1: Load your trained model
model = RTMDet(
    model="user_config.toml",      # Config generated during training
    checkpoint_path="epoch_100.pth" # Your trained checkpoint
)

# Option 2: Use pretrained model
model = RTMDet("rtmdet_s")  # Auto-downloads pretrained weights

# Run inference
results = model.predict("sample.jpg", show=True)

# Access clean, structured results
for box in results[0].boxes:
    print(f"Class: {box.cls}, Score: {box.conf:.3f}, BBox: {box.xyxy}")
```

![Inference Demo](docs/inference.gif)

---

## 🚢 3. Export

Deploying to production with mmdeploy is usually a nightmare. We simplified it to one command using **MMDeploy via Docker**.

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

## 🧘 Custom Pose Estimation? Still EZ.

Training on custom keypoints? Just add your metainfo to the TOML. **You can add as many keypoints as your dataset requires.**

```toml
# pose_dataset.toml
data_root = "datasets/custom_pose"
classes = ["dog"]

[train]
ann_file = "annotations/train.json"
img_dir = "images/train"

[val]
ann_file = "annotations/val.json"
img_dir = "images/val"

[metainfo]
sigmas = [0.025, 0.025, 0.05]  # One per keypoint
joint_weights = [1.0, 1.0, 1.0] # One per key point

[metainfo.keypoint_info.0]
name = "nose"
id = 0
color = [51, 153, 255]

[metainfo.keypoint_info.1]
name = "left_eye"
id = 1
color = [51, 153, 255]

[metainfo.keypoint_info.2]
name = "right_eye"
id = 2
color = [51, 153, 255]

[metainfo.skeleton_info.0]
link = ["nose", "left_eye"]
id = 0

[metainfo.skeleton_info.1]
link = ["nose", "right_eye"]
id = 1

# ADD AS MANY KEYPOINT INFO AS YOU NEED

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

### Requirements

- Python 3.9 or 3.10
- NVIDIA GPU with CUDA 11.7 (for GPU version)
- Linux or Windows
- Git

### Quick Install (Recommended)

**GPU (CUDA 11.7):**

```bash
curl -sSL https://raw.githubusercontent.com/JustAnalyze/ez_openmmlab/main/install.sh | bash
```

**CPU:**

```bash
curl -sSL https://raw.githubusercontent.com/JustAnalyze/ez_openmmlab/main/install-cpu.sh | bash
```

### Manual Installation

#### GPU (CUDA 11.7)

```bash
# Step 1: Install PyTorch with CUDA support
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu117

# Step 2: Install MMCV with CUDA support
pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html

# Step 3: Install ez-openmmlab
pip install ez-openmmlab
```

#### CPU Only

```bash
# Step 1: Install PyTorch (CPU)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install MMCV (CPU)
pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html

# Step 3: Install ez-openmmlab
pip install ez-openmmlab
```

---

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
