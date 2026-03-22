from ez_openmmlab import RTMDet

# provide the user_config.toml of the unfinished training
model = RTMDet(model="path/to/user_config.toml")

model.resume(epochs=150)
