# 🚀 ez_openmmlab: OpenMMLab Made EZ

Utilize OpenMMLab using an EZ and Familiar API ;)

`ez_openmmlab` is a high-level, **TOML-first** wrapper that makes SOTA models like **RTMDet**, **RTMPose**, and **RTMO** actually EZ to use. Stop fighting with 500-line Python configs and dataset registries—just write a few lines of TOML and get back to building.

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
One method. All the power.

```python
from ez_openmmlab import RTMDet

# Initialize (choices: rtmdet_tiny, rtmdet_s, rtmpose_m, etc.)
model = RTMDet("rtmdet_tiny")

# Start training. No more config file surgeries!
model.train(
    dataset_config_path="dataset.toml",
    epochs=100,
    batch_size=16,
    scale_factor=(0.5, 2.0), # Example: Data augmentation is EZ too
    random_flip_prob=0.5
)
```
![Training Demo](docs/train.gif)

---

## 🔍 2. Inference
Predict and visualize results with a single line.

```python
# Inference made simple
results = model.predict("sample.jpg", show=True)

# Access clean, structured results
for box in results[0].boxes:
    print(f"Class: {box.cls}, Score: {box.conf}, BBox: {box.xyxy}")
```
![Inference Demo](docs/inference.gif)

---

## 🚢 3. Export
Deploying to production is usually a nightmare. We simplified it to one command using **MMDeploy via Docker**.

```python
model.export(
    format="onnx",        # Target format: 'onnx' or 'tensorrt'
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
# dataset.toml (Pose Version)
[metainfo]
sigmas = [0.025, 0.025, 0.05] # One per keypoint

[metainfo.keypoint_info.0]
name = "nose"
id = 0
color = [51, 153, 255]

[metainfo.skeleton_info.0]
link = ["nose", "left_eye"]
id = 0
```

```python
from ez_openmmlab import RTMPose
model = RTMPose("rtmpose_s")
model.train(dataset_config_path="pose_data.toml", epochs=210)
```

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/JustAnalyze/ez_openmmlab.git
cd ez_openmmlab

# 2. Run the EZ Installer
# This handles uv, submodules, PyTorch, and all build dependencies.
chmod +x install.sh
./install.sh
```

---

## ✨ Key Features

- **EZ Environment:** Reproducible setups that just work via `uv`.
- **EZ Configuration:** Human-readable TOML replaces complex Python config inheritance.
- **Auto-Magic Checkpoints:** Missing weights? We download them for you automatically.
- **Strict Validation:** Powered by Pydantic to catch errors _before_ you start your run.
- **Performance Optimized:** Vectorized, NumPy-first results with **Lazy Initialization**.

---

## 🗺️ Roadmap

- [x] **Native Export:** One-click `.export()` to ONNX and TensorRT.
- [ ] **Full CLI:** Run training and inference directly from your terminal.
- [ ] **Architecture Expansion:** Bringing the "EZ" treatment to more OpenMMLab models.

---

## 🤝 Acknowledgements

`ez_openmmlab` wouldn't exist without the relentless research and engineering of the **OpenMMLab** team.

**Currently Supported:**
- **Detection & Segmentation:** `rtmdet` (all variants)
- **2D Pose Estimation:** `rtmpose`, `rtmo`
