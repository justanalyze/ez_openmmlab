from ez_openmmlab import RTMDet

# 1. Initialize Instance Segmentation model (rtmdet-ins_tiny, rtmdet-ins_s, etc.)
model = RTMDet("rtmdet-ins_tiny")

# 2. Predict with masks (segmentation is automatic for -ins models)
results = model.predict(
    image_path="demos/demo.jpg",
    device="cpu", # Change to "cuda" if you have a GPU
    show=True,
    out_dir="runs/rtmdet_ins_demo",
)

# 3. Explore the results
result = results[0]
print(f"Detected {len(result.masks)} segmentations.")

for i in range(len(result.masks)):
    label = result.names[int(result.boxes.cls[i])]
    conf = result.boxes.conf[i]
    # result.masks provides easy access to segmentation data
    mask_shape = result.masks.data[i].shape
    print(f"- {label} ({conf:.2f}): Mask of shape {mask_shape}")
    # You can access mask points via result.masks.xy[i]
