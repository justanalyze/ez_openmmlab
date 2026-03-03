from ez_openmmlab import RTMDet

# 1. Initialize the model (options: rtmdet_tiny, rtmdet_s, rtmdet_m, etc.)
model = RTMDet("rtmdet_tiny")

# 2. Run inference (supports image path, list of paths, or directory)
results = model.predict(
    image_path="demos/demo.jpg",
    device="cpu", # Change to "cuda" if you have a GPU
    show=True,
    out_dir="runs/rtmdet_demo",
)

# 3. Access structured results easily
result = results[0]
print(f"Detected {len(result.boxes)} objects.")

for i in range(len(result.boxes)):
    label = result.names[int(result.boxes.cls[i])]
    conf = result.boxes.conf[i]
    bbox = result.boxes.xyxy[i]
    print(f"- {label} ({conf:.2f}): {bbox}")
