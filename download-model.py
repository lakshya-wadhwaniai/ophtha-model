from truefoundry.ml import get_client

# No need to pass the token, as it will be read from the environment variable
client = get_client()

# Get the model version by its FQN
model_version = client.get_model_version_by_fqn("model:wadhwaniai/ophtha-deployment/multiclass-efficientnetv2:5")

# Download the model to the specified location
download_info = model_version.download(path="/Users/lakshyavijh/Downloads/package/inference-model")

# Optionally, you can print or inspect the download_info if needed
print(f"Model downloaded to: {download_info.model_dir}")
