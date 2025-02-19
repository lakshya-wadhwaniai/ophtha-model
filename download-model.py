from truefoundry.ml import get_client

# No need to pass the token, as it will be read from the environment variable
client = get_client()

# Get the inference model version by its FQN and download 
# model_version = client.get_model_version_by_fqn("model:wadhwaniai/ophtha-deployment/multiclass-efficientnetv2:5")
# download_info = model_version.download(path="./inference-model")

# Get the gradability model version by its FQN and download 
model_version = client.get_model_version_by_fqn("model:wadhwaniai/ophtha-deployment/gradability-efficientnetv2:4")

download_info = model_version.download(path="./gradability-model")

# Optionally, you can print or inspect the download_info if needed
print(f"Model downloaded to: {download_info.model_dir}")