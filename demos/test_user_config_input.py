from ez_openmmlab import RTMDet

model = RTMDet(
    model="./runs/demo_rtmdet/user_config.toml",
    checkpoint_path="./runs/demo_rtmdet/epoch_5.pth",
)

results = model.predict(
    "demos/demo.jpg", device="cpu", out_dir="runs/test_user_config_input"
)
print(results[0].boxes.xyxy[0])
