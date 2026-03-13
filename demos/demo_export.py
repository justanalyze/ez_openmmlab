from pathlib import Path
from ez_openmmlab import RTMDet

# 1. Provide the paths to your custom configuration and weights
# For production export, we force you to be explicit about the weights
# to ensure the exported model matches exactly what you intended.
config_path = "runs/rtmdet_sample_training_v3/user_config.toml"
checkpoint_path = "runs/rtmdet_sample_training_v3/epoch_10.pth"  # Be explicit!

# 2. Initialize the model
model = RTMDet(model=config_path, checkpoint_path=checkpoint_path)

# 3. Run Export
# The export method will:
# - Flatten the config into a single file
# - Select a 'Dynamic' deployment recipe (Flexible image sizes)
# - Run MMDeploy inside Docker to produce the production artifact
print(f"Exporting model with weights: {checkpoint_path}...")

export_dir = model.export(
    format="onnx",
    image="demos/demo.jpg",  # Used to 'trace' the model graph
    output_dir="runs/export_production",
    device="cpu",
)

print(f"Success! Production model saved to: {export_dir}")
