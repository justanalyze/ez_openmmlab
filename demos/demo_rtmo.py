from ez_openmmlab import RTMO

# 1. RTMO is a BOTTOM-UP one-stage model (no separate detector needed)
model = RTMO("rtmo_s")

# 2. Predict image with pose estimation
results = model.predict(
    image_path="demos/demo.jpg",
    device="cpu", # Change to "cuda" if you have a GPU
    show=True,
    out_dir="runs/rtmo_demo",
)

# 3. Access keypoints results
result = results[0]
print(f"Detected pose for {len(result.keypoints)} people.")

for i in range(len(result.keypoints)):
    # Keypoints data available as xy coords and confidence
    kpts = result.keypoints.xy[i]
    kpts_conf = result.keypoints.conf[i]
    print(f"Person {i}: {len(kpts)} keypoints, average confidence: {kpts_conf.mean():.2f}")
