# 🧘 Full Pose Configuration Guide

This guide walks you through everything you need to train, run inference,
and export a pose estimation model using `ez_openmmlab` — from setting up
your COCO annotations to picking the right model for your use case.

---

## Table of Contents

- [Supported Models](#-supported-models)
- [Understanding COCO Pose Annotations](#-understanding-coco-pose-annotations)
- [Dataset Configuration (pose_dataset.toml)](#-dataset-configuration-pose_datasettoml)
- [Training](#-training)
- [Inference](#-inference)
- [Export](#-export)
- [Common Mistakes](#-common-mistakes)
- [Choosing the Right Model](#-choosing-the-right-model)

---

## 🤖 Supported Models

### RTMPose — Single-Person & Multi-Person Pose Estimation
Best for scenarios where you need accurate keypoint detection per individual.

| Model | Speed | Accuracy | Best For |
| :--- | :--- | :--- | :--- |
| `rtmpose_tiny` | ⚡⚡⚡⚡ | ⭐⭐ | Edge devices, real-time on CPU |
| `rtmpose_s` | ⚡⚡⚡ | ⭐⭐⭐ | Balanced speed and accuracy |
| `rtmpose_m` | ⚡⚡ | ⭐⭐⭐⭐ | Production use cases |
| `rtmpose_l` | ⚡ | ⭐⭐⭐⭐⭐ | Highest accuracy, GPU recommended |

### RTMO — Real-Time Multi-Person One-Stage Pose Estimation
Best for scenes with many people — detects and estimates pose in a single pass.

| Model | Speed | Accuracy | Best For |
| :--- | :--- | :--- | :--- |
| `rtmo_s` | ⚡⚡⚡ | ⭐⭐⭐ | Crowded scenes, faster inference |
| `rtmo_m` | ⚡⚡ | ⭐⭐⭐⭐ | Balanced for multi-person |
| `rtmo_l` | ⚡ | ⭐⭐⭐⭐⭐ | Best multi-person accuracy |

> **RTMPose vs RTMO — which should I use?**
> Use **RTMPose** when subjects are already cropped or isolated (e.g. one
> animal per image). Use **RTMO** when your images contain multiple subjects
> and you want detection + pose in a single forward pass.

---

## 📋 Understanding COCO Pose Annotations

Before configuring your dataset, your annotation JSON must follow the
COCO keypoint format. Here is the minimum structure `ez_openmmlab` expects:

```json
{
  "categories": [
    {
      "id": 1,
      "name": "your_class",
      "keypoints": ["keypoint_0", "keypoint_1", "keypoint_2"],
      "skeleton": [[0, 1], [1, 2]]
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "keypoints": [x0, y0, v0, x1, y1, v1, x2, y2, v2],
      "num_keypoints": 3,
      "bbox": [x, y, width, height],
      "area": 1234.5,
      "iscrowd": 0
    }
  ],
  "images": [
    {
      "id": 1,
      "file_name": "image_001.jpg",
      "width": 640,
      "height": 480
    }
  ]
}
```

### Keypoint Visibility Flags

Each keypoint has three values: `x, y, v` where `v` is the visibility flag:

| Value | Meaning |
| :--- | :--- |
| `0` | Not labeled — keypoint does not exist in this annotation |
| `1` | Labeled but not visible — occluded or outside frame |
| `2` | Labeled and visible |

### Example — 3 Keypoint Animal

```json
"keypoints": ["head", "body", "tail"],
"skeleton": [[0, 1], [1, 2]],

"keypoints": [120, 85, 2, 145, 130, 2, 200, 180, 1]
```
This means: head at (120, 85) visible, body at (145, 130) visible,
tail at (200, 180) occluded.

---

## ⚙️ Dataset Configuration (pose_dataset.toml)

Your `pose_dataset.toml` must mirror your COCO annotation file exactly.
The order and names of keypoints and skeleton connections must match.

### Minimal Example — 3 Keypoints

```toml
dataset_name = "MY_POSE_DATASET"
data_root = "datasets/my_pose_project"
classes = ["your_class"]

[train]
ann_file = "annotations/train.json"
img_dir = "images/train"

[val]
ann_file = "annotations/val.json"
img_dir = "images/val"

[metainfo]
# One sigma value per keypoint — controls the OKS evaluation metric
# Smaller value = stricter scoring for that keypoint
# Typical ranges: 0.025 (precise points like eyes) to 0.1 (loose points like hips)
sigmas = [0.05, 0.05, 0.05]

# One weight per keypoint — higher weight = more training emphasis
joint_weights = [1.0, 1.0, 1.0]

# --- Keypoint definitions ---
# id must start at 0 and match the index in your COCO "keypoints" list

[metainfo.keypoint_info.0]
name = "head"        # Must match COCO keypoints[0]
id = 0
color = [51, 153, 255]

[metainfo.keypoint_info.1]
name = "body"        # Must match COCO keypoints[1]
id = 1
color = [0, 255, 0]

[metainfo.keypoint_info.2]
name = "tail"        # Must match COCO keypoints[2]
id = 2
color = [255, 128, 0]

# --- Skeleton connections ---
# link names must match keypoint names defined above
# id must start at 0

[metainfo.skeleton_info.0]
link = ["head", "body"]    # Must match COCO skeleton[0]
id = 0
color = [0, 255, 0]

[metainfo.skeleton_info.1]
link = ["body", "tail"]    # Must match COCO skeleton[1]
id = 1
color = [255, 128, 0]
```

### Extended Example — 17 Keypoints (Human Body)

```toml
dataset_name = "HUMAN_POSE_DATASET"
data_root = "datasets/human_pose"
classes = ["person"]

[train]
ann_file = "annotations/train.json"
img_dir = "images/train"

[val]
ann_file = "annotations/val.json"
img_dir = "images/val"

[metainfo]
sigmas = [
  0.026, 0.025, 0.025, 0.035, 0.035,
  0.079, 0.079, 0.072, 0.072, 0.062,
  0.062, 0.107, 0.107, 0.087, 0.087,
  0.089, 0.089
]
joint_weights = [
  1.0, 1.0, 1.0, 1.0, 1.0,
  1.2, 1.2, 1.5, 1.5, 1.5,
  1.5, 1.0, 1.0, 1.2, 1.2,
  1.5, 1.5
]

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

[metainfo.keypoint_info.3]
name = "left_ear"
id = 3
color = [51, 153, 255]

[metainfo.keypoint_info.4]
name = "right_ear"
id = 4
color = [51, 153, 255]

[metainfo.keypoint_info.5]
name = "left_shoulder"
id = 5
color = [0, 255, 0]

[metainfo.keypoint_info.6]
name = "right_shoulder"
id = 6
color = [255, 128, 0]

[metainfo.keypoint_info.7]
name = "left_elbow"
id = 7
color = [0, 255, 0]

[metainfo.keypoint_info.8]
name = "right_elbow"
id = 8
color = [255, 128, 0]

[metainfo.keypoint_info.9]
name = "left_wrist"
id = 9
color = [0, 255, 0]

[metainfo.keypoint_info.10]
name = "right_wrist"
id = 10
color = [255, 128, 0]

[metainfo.keypoint_info.11]
name = "left_hip"
id = 11
color = [0, 255, 0]

[metainfo.keypoint_info.12]
name = "right_hip"
id = 12
color = [255, 128, 0]

[metainfo.keypoint_info.13]
name = "left_knee"
id = 13
color = [0, 255, 0]

[metainfo.keypoint_info.14]
name = "right_knee"
id = 14
color = [255, 128, 0]

[metainfo.keypoint_info.15]
name = "left_ankle"
id = 15
color = [0, 255, 0]

[metainfo.keypoint_info.16]
name = "right_ankle"
id = 16
color = [255, 128, 0]

[metainfo.skeleton_info.0]
link = ["left_ankle", "left_knee"]
id = 0
color = [0, 255, 0]

[metainfo.skeleton_info.1]
link = ["left_knee", "left_hip"]
id = 1
color = [0, 255, 0]

[metainfo.skeleton_info.2]
link = ["right_ankle", "right_knee"]
id = 2
color = [255, 128, 0]

[metainfo.skeleton_info.3]
link = ["right_knee", "right_hip"]
id = 3
color = [255, 128, 0]

[metainfo.skeleton_info.4]
link = ["left_hip", "right_hip"]
id = 4
color = [51, 153, 255]

[metainfo.skeleton_info.5]
link = ["left_shoulder", "left_hip"]
id = 5
color = [51, 153, 255]

[metainfo.skeleton_info.6]
link = ["right_shoulder", "right_hip"]
id = 6
color = [51, 153, 255]

[metainfo.skeleton_info.7]
link = ["left_shoulder", "right_shoulder"]
id = 7
color = [51, 153, 255]

[metainfo.skeleton_info.8]
link = ["left_shoulder", "left_elbow"]
id = 8
color = [0, 255, 0]

[metainfo.skeleton_info.9]
link = ["right_shoulder", "right_elbow"]
id = 9
color = [255, 128, 0]

[metainfo.skeleton_info.10]
link = ["left_elbow", "left_wrist"]
id = 10
color = [0, 255, 0]

[metainfo.skeleton_info.11]
link = ["right_elbow", "right_wrist"]
id = 11
color = [255, 128, 0]

[metainfo.skeleton_info.12]
link = ["left_eye", "right_eye"]
id = 12
color = [51, 153, 255]

[metainfo.skeleton_info.13]
link = ["nose", "left_eye"]
id = 13
color = [51, 153, 255]

[metainfo.skeleton_info.14]
link = ["nose", "right_eye"]
id = 14
color = [51, 153, 255]

[metainfo.skeleton_info.15]
link = ["left_eye", "left_ear"]
id = 15
color = [51, 153, 255]

[metainfo.skeleton_info.16]
link = ["right_eye", "right_ear"]
id = 16
color = [51, 153, 255]

[metainfo.skeleton_info.17]
link = ["left_ear", "left_shoulder"]
id = 17
color = [51, 153, 255]

[metainfo.skeleton_info.18]
link = ["right_ear", "right_shoulder"]
id = 18
color = [51, 153, 255]
```

---

## 🏋️ Training

```python
from ez_openmmlab import RTMPose

# Initialize with your chosen model size
model = RTMPose("rtmpose_s")

model.train(
    dataset_config_path="pose_dataset.toml",
    epochs=210,
    batch_size=32,
)
```

> [!TIP]
> Training interrupted? Resume instantly:
> ```python
> model = RTMPose(model="user_config.toml")
> model.resume()
> ```

> [!IMPORTANT]
> **Windows users:** Always wrap in `if __name__ == "__main__":` to avoid
> multiprocessing errors with PyTorch's DataLoader.

### Using RTMO

```python
from ez_openmmlab import RTMO

model = RTMO("rtmo_m")
model.train(
    dataset_config_path="pose_dataset.toml",
    epochs=210,
    batch_size=32,
)
```

### Recommended Epochs by Model

| Model | Recommended Epochs | Notes |
| :--- | :--- | :--- |
| `rtmpose_tiny` | 150–210 | Converges faster |
| `rtmpose_s` | 210 | Good baseline |
| `rtmpose_m` | 210–300 | More data benefits longer runs |
| `rtmpose_l` | 300+ | GPU recommended |
| `rtmo_s/m/l` | 210–300 | Multi-person needs more examples |

---

## 🔍 Inference

```python
from ez_openmmlab import RTMPose

# Load your trained model
model = RTMPose(
    model="user_config.toml",
    checkpoint_path="best_model.pth"
)

# Or use a pretrained model directly
model = RTMPose("rtmpose_m")

# Run inference
results = model.predict("image.jpg", show=True)

# Access structured results
for person in results[0].keypoints:
    print(f"Keypoint coordinates: {person.xy}")       # shape: [num_keypoints, 2]
    print(f"Keypoint scores:      {person.conf}")     # shape: [num_keypoints]
    print(f"Visible keypoints:    {person.visible}")  # shape: [num_keypoints]
```

### Batch Inference

```python
import glob

image_paths = glob.glob("images/*.jpg")
results = model.predict(image_paths, show=False)

for i, result in enumerate(results):
    for person in result.keypoints:
        print(f"Image {i} — keypoints: {person.xy}")
```

---

## 🚢 Export

> [!IMPORTANT]
> **Docker is required** for export. The MMDeploy image will be downloaded
> on first use (warning: 30GB+).

```python
from ez_openmmlab import RTMPose

model = RTMPose(
    model="user_config.toml",
    checkpoint_path="best_model.pth"
)

# Export to ONNX (CPU or GPU)
model.export(
    format="onnx",
    image="sample.jpg",
    output_dir="deploy/",
    device="cpu"
)

# Export to TensorRT (GPU only)
model.export(
    format="tensorrt",
    image="sample.jpg",
    output_dir="deploy/",
    device="cuda"
)
```

---

## ⚠️ Common Mistakes

### Keypoint order mismatch
Your `pose_dataset.toml` keypoint order must exactly match your COCO JSON.

```json
// COCO JSON
"keypoints": ["head", "body", "tail"]
```

```toml
# CORRECT — same order
[metainfo.keypoint_info.0]
name = "head"

[metainfo.keypoint_info.1]
name = "body"

[metainfo.keypoint_info.2]
name = "tail"
```

```toml
# WRONG — different order, will cause silent errors
[metainfo.keypoint_info.0]
name = "tail"
```

---

### Wrong number of sigmas or joint_weights
You must have exactly one value per keypoint. Mismatches cause an error at
training start.

```toml
# 3 keypoints defined → must have exactly 3 values
sigmas = [0.05, 0.05, 0.05]        # ✅ correct
joint_weights = [1.0, 1.0, 1.0]    # ✅ correct

sigmas = [0.05, 0.05]              # ❌ wrong — only 2 values for 3 keypoints
```

---

### Skeleton link names don't match keypoint names

```toml
[metainfo.keypoint_info.0]
name = "head"   # defined as "head"

[metainfo.skeleton_info.0]
link = ["Head", "body"]   # ❌ wrong — capital H won't match "head"
link = ["head", "body"]   # ✅ correct — exact match, case sensitive
```

---

### Classes list doesn't match COCO categories

```json
// COCO JSON
"categories": [{"id": 1, "name": "crayfish"}]
```

```toml
classes = ["crayfish"]   # ✅ correct
classes = ["Crayfish"]   # ❌ wrong — case sensitive
classes = ["shrimp"]     # ❌ wrong — name doesn't match
```

---

## 🎯 Choosing the Right Model

| Scenario | Recommended Model |
| :--- | :--- |
| One subject per image, fast inference needed | `rtmpose_tiny` or `rtmpose_s` |
| One subject per image, accuracy matters | `rtmpose_m` or `rtmpose_l` |
| Multiple subjects per image | `rtmo_s`, `rtmo_m`, or `rtmo_l` |
| Edge device / CPU only deployment | `rtmpose_tiny` or `rtmo_s` |
| Research or highest possible accuracy | `rtmpose_l` or `rtmo_l` |
| Custom animal keypoints (few keypoints) | `rtmpose_s` — trains fast |
| Custom animal keypoints (many keypoints) | `rtmpose_m` — more capacity |

---

## 📖 More Resources

- [Demo Examples](../demos/) — End-to-end working examples
- [Main README](../README.md) — Back to the main documentation
- [Issues](https://github.com/JustAnalyze/ez_openmmlab/issues) — Report
  bugs or request features
