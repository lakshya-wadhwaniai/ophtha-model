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
# from tritonclient.http.auth import BasicAuth
# from PIL import Image, ImageFilter
# from PIL import ImageDraw, ImageChops
import json
from preprocess import padded_crop

app = FastAPI(root_path=os.getenv("TFY_SERVICE_ROOT_PATH"))
url = os.getenv("TRITONSERVER_URL")
ssl = (os.getenv("SSL").lower() == 'true')
print(type(ssl))
print(ssl)

@app.post("/upload/image/")
async def upload_image(image: UploadFile = File(...), model_name: str = Form(...)) -> Dict:
    # try:
    assert model_name in ['classification_model', 'gradability_model']
    # print(username, password)
    contents = await image.read()

    im = np.frombuffer(contents, np.uint8)
    im = cv2.imdecode(im, cv2.IMREAD_COLOR)
    im = padded_crop(im)

    #save it to s3
    time.sleep(5)

    results = {

    }
    return results
    
    # return tritonserver(im, url, model_name)

    # except Exception as e:
    #     return {"new error": str(e)}


def tritonserver(image_data, url, model_name):
    tick = time.time()
    try:
        triton_client = httpclient.InferenceServerClient(url = url, ssl=ssl)

    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)
    
    inputs = []
    outputs0 = []
    outputs1 = []
    input_name = "raw_image"
    output0_name = "output_0"
    output1_name = "output_1"

    image_data = np.expand_dims(image_data, axis=0)

    inputs.append(
        httpclient.InferInput(input_name, image_data.shape, "UINT8"))

    outputs = [(httpclient.InferRequestedOutput(output0_name))]

    if model_name == 'classification_model':
        name_to_code_mapping = {
                "No Diabetic Retinopathy": 0,
                "Mild Diabetic Retinopathy": 1,
                "Moderate Diabetic Retinopathy": 2,
                "Severe Diabetic Retinopathy": 3,
                "Proliferative Diabetic Retinopathy": 4,
            }

        code_to_name_mapping = {v: k for k, v in name_to_code_mapping.items()}

    elif model_name == 'gradability_model':
        name_to_code_mapping = {
                "Gradable": 0,
                "Non Gradable": 1,
            }

        code_to_name_mapping = {v: k for k, v in name_to_code_mapping.items()}


    inputs[0].set_data_from_numpy(image_data)
    results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)

    output0_data = results.as_numpy(output0_name)[0]
    output1_data = results.as_numpy(output0_name)[1]

    if model_name == 'gradability_model':
        output0_data = output0_data[0]
        output1_data = output1_data[0]


    print(output0_data)
    # print(output1_data)
    print(code_to_name_mapping[output1_data])
    tock = time.time()
    print("Time to run inference: ", tock-tick)


    results = {
        'model_name' : "EfficientNet-V2M",
        'model_version' : "v0.1.1",
        'model_score' : float(output0_data),
        'predicted_class' : float(output1_data),
        'predicted_class_name' : code_to_name_mapping[output1_data],
        'inference_time': float(tock-tick)
    }
    return results


