from fastapi import FastAPI, File, UploadFile, Form
from io import BytesIO
from typing import Dict
import uvicorn
import os
import numpy as np
import time
import sys
from tritonclient.utils import *
import tritonclient.http as httpclient
import cv2
import json
from preprocess import padded_crop
import logging

app = FastAPI(root_path=os.getenv("TFY_SERVICE_ROOT_PATH"))
url = os.getenv("TRITONSERVER_URL")
print("url", url)
ssl = (os.getenv("SSL").lower() == 'true')

# Set up logging
logging.basicConfig(level=logging.INFO)


@app.post("/upload/image/")
async def upload_image(image: UploadFile = File(...), model_name: str = Form(...)) -> Dict:
    try:
        assert model_name in ['classification_model', 'gradability_model']
        contents = await image.read()

        im = np.frombuffer(contents, np.uint8)
        im = cv2.imdecode(im, cv2.IMREAD_COLOR)
        im = padded_crop(im)  # ensure this is handling errors gracefully
        
        # Perform inference
        results = await tritonserver(im, url, model_name)
        return results
    except Exception as e:
        logging.error(f"Error during image upload and inference: {e}")
        return {"error": "An error occurred during processing the image."}

async def tritonserver(image_data, url, model_name):
    tick = time.time()
    try:
        triton_client = httpclient.InferenceServerClient(url=url, ssl=ssl)
    except Exception as e:
        logging.error(f"Error creating Triton client: {e}")
        return {"error": "Error connecting to the Triton server."}
    
    inputs = []
    outputs0 = []
    outputs1 = []
    input_name = "raw_image"
    output0_name = "output_0"
    output1_name = "output_1"

    image_data = np.expand_dims(image_data, axis=0)
    inputs.append(httpclient.InferInput(input_name, image_data.shape, "UINT8"))
    outputs = [(httpclient.InferRequestedOutput(output0_name))]

    # Handle model-specific output names and mapping
    name_to_code_mapping = {}
    if model_name == 'classification_model':
        name_to_code_mapping = {
            0: "No Diabetic Retinopathy",
            1: "Mild Diabetic Retinopathy",
            2: "Moderate Diabetic Retinopathy",
            3: "Severe Diabetic Retinopathy",
            4: "Proliferative Diabetic Retinopathy",
        }
    elif model_name == 'gradability_model':
        name_to_code_mapping = {
            0:"Gradable",
            1:"Non Gradable",
        }

    inputs[0].set_data_from_numpy(image_data)
    try:
        results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        return {"error": f"Inference failed: {str(e)}"}
    output0_data = results.as_numpy(output0_name)[0]
    output1_data = results.as_numpy(output0_name)[1]
    if model_name == 'gradability_model':
        output0_data = output0_data[0]
        output1_data = output1_data[0]

    tock = time.time()
    logging.info(f"Time to run inference: {tock - tick}")
    rounded_output = round(float(output1_data))
    predicted_class_name = name_to_code_mapping.get(rounded_output, "Unknown")


    return {
        'model_name': "EfficientNet-V2M",
        'model_version': "v0.1.4",
        'model_score': float(output0_data),
        'predicted_class': float(output1_data),
        'predicted_class_name': predicted_class_name,
        'inference_time': float(tock - tick)
    }
