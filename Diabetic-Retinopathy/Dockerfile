FROM nvcr.io/nvidia/tritonserver:23.01-py3

# Update system and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/app

# Copy and install Python dependencies (if required)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --timeout=120 -r requirements.txt

# Copy all project files
COPY . /opt/tritonserver

# Set the correct working directory
WORKDIR /opt/tritonserver

# Copy model repository (make sure this folder exists in your project)
COPY model_repository /model_repository


# Expose the Triton Inference Server port
EXPOSE 8000

# Run Triton Inference Server with model repository
# CMD ["tritonserver", "--model-repository=/model_repository"]

CMD ["tritonserver", "--model-repository=/model_repository", "--log-verbose=1", "--log-file=/opt/tritonserver/triton.log"]

