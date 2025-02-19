# ophtha-model

This project consists of two servers deployed using Docker Compose:

1. **Inference Server (`ophtha-inference`)** – Runs using Triton Inference Server.
2. **FastAPI Server (`ophtha-fastapi`)** – Acts as a client application interacting with the inference server.

## Model Files

The model files are located in the following locations:
- Local server:
  - `/home/ubuntu/gradability-model/data/model.pth`
  - `/home/ubuntu/inference-model/data/model.pth`
- Remote S3 Storage:
  - `arn:aws:s3:::ophtha-model-prod`

These model files need to be copied to the respective directories inside the repository:
- `ophtha-model/Diabetic-Retinopathy/model_repository/gradability_model/1`
- `ophtha-model/Diabetic-Retinopathy/model_repository/classification_model/1`

## Deployment Setup

### ** Way to Deploy**
**Build Docker images from source code**

#### **1. Inference Server (`ophtha-inference`):**
- Built from the `Diabetic-Retinopathy` directory.
- Uses the `Dockerfile` present in that directory.
- Exposes the Triton server on port `8000`.

```yaml
ophtha-inference:
    build:
      context: ./Diabetic-Retinopathy  # Path to the directory where the Triton Dockerfile is located
      dockerfile: Dockerfile
    ports:
      - "8000:8000"  # Expose Triton server on port 8000
    networks:
      - triton_net
```

#### **2. FastAPI Server (`ophtha-fastapi`):**
- Built from the `Diabetic-Retinopathy/src/app` directory.
- Uses `Dockerfile_clientapp`.
- Exposes the FastAPI server on port `8001`.
- Uses an `.env` file for environment variables.
- Depends on the `ophtha-inference` server to ensure the inference server starts first.

```yaml
ophtha-fastapi:
    build:
      context: ./Diabetic-Retinopathy/src/app  # Path to the FastAPI Dockerfile directory
      dockerfile: Dockerfile_clientapp
    ports:
      - "8001:8001"  # Expose FastAPI on port 8001
    networks:
      - triton_net
    env_file:
      - .env  # Load environment variables from your .env file
    depends_on:
      - ophtha-inference  # Ensure Triton starts before FastAPI
```

## **How to Deploy**

### Build from Source**

1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd ophtha-model
   ```

2. **Build the Docker images:**
   ```sh
   docker compose build
   ```

3. **Start the containers:**
   ```sh
   docker compose up -d
   ```

## **Stopping the Servers**
To stop the running services, use:
```sh
docker compose down
```

## **Accessing the Servers**
- Triton Inference Server: `http://localhost:8000`
- FastAPI Server: `http://localhost:8001`



