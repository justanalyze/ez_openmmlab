from ez_openmmlab import RTMDet

model = RTMDet(model="./runs/rtmdet_sample_training_v2/user_config.toml")

model.resume(epochs=150)
