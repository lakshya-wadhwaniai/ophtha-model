# ophtha-model

This project consists of two servers deployed using Docker Compose:

1. **Inference Server (`ophtha-inference`)** – Runs using Triton Inference Server.
2. **FastAPI Server (`ophtha-fastapi`)** – Acts as a client application interacting with the inference server.

## Deployment Setup

### **1. Inference Server (`ophtha-inference`):**
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
```

### **2. FastAPI Server (`ophtha-fastapi`):**
- Built from the `Diabetic-Retinopathy/src/app` directory.
- Uses `Dockerfile_clientapp`.
- Exposes the FastAPI server on port `8001`.
- Uses an `.env` file for environment variables.
- Depends on the `ophtha-inference` server to ensure the inference server starts first.

```yaml
optha-fastapi:
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

To deploy both services, simply run the following commands:

1. **Build the Docker images:**
   ```sh
   docker-compose build
   ```

2. **Start the containers:**
   ```sh
   docker-compose up -d
   ```

This will spin up both the inference server and the FastAPI server.

## **Stopping the Servers**
To stop the running services, use:
```sh
docker-compose down
```

## **Accessing the Servers**
- Triton Inference Server: `http://localhost:8000`
- FastAPI Server: `http://localhost:8001`

Let me know if you need any modifications! 🚀

