from ez_openmmlab import RTMDet

model = RTMDet(
    model="./runs/demo_rtmdet/user_config.toml",
    checkpoint_path="./runs/demo_rtmdet/epoch_5.pth",
)

results = model.predict(
    "demos/demo.jpg", device="cpu", out_dir="runs/test_user_config_input"
)

for result in results:
    for bbox in result.boxes:
        if bbox.conf > 0.6:
            print(bbox.conf)
            print(bbox.xyxy)
            print(bbox.cls)
