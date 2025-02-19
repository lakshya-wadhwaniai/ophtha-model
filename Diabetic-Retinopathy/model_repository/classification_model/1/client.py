import numpy as np
import time
import sys
from tritonclient.utils import *
import tritonclient.http as httpclient
import argparse
from PIL import Image, ImageFilter
from PIL import ImageDraw, ImageChops
import json

def load_image(fname: str, crop_size=512):
    im = Image.open(fname).convert('RGB')
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -30)
    bbox = diff.getbbox()
    if bbox:
        im = im.crop(bbox)
    height,width = im.size
    lum_img = Image.new('L', [height,width] , 0)
    draw = ImageDraw.Draw(lum_img)
    draw.pieslice([(0,0), (height,width)], 0, 360, fill = 255, outline = "white")
    img_arr =np.array(im)
    lum_img_arr =np.array(lum_img)
    
    final_img_arr = np.dstack((img_arr,lum_img_arr))
    im = (Image.fromarray(final_img_arr)).resize([crop_size, crop_size], Image.Resampling.LANCZOS).convert('RGB')

    return np.asarray(im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",
                        type=str,
                        required=True,
                        help="Path to the image")
    parser.add_argument("--url",
                        type=str,
                        required=False,
                        default="localhost:8000",
                        help="Inference server URL. Default is localhost:8000.")
    parser.add_argument('-v',
                        "--verbose",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    args = parser.parse_args()

    try:
        triton_client = httpclient.InferenceServerClient(url = args.url, ssl=True)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    inputs = []
    outputs0 = []
    outputs1 = []
    input_name = "raw_image"
    output0_name = "output_0"
    output1_name = "output_1"

    print(args.image)

    image_data = load_image(args.image)
    image_data = np.expand_dims(image_data, axis=0)

    inputs.append(
        httpclient.InferInput(input_name, image_data.shape, "UINT8"))

    outputs = [(httpclient.InferRequestedOutput(output0_name))]


    name_to_code_mapping = {
            "No Diabetic Retinopathy": 0,
            "Mild Diabetic Retinopathy": 1,
            "Moderate Diabetic Retinopathy": 2,
            "Severe Diabetic Retinopathy": 3,
            "Proliferative Diabetic Retinopathy": 4,
        }

    code_to_name_mapping = {v: k for k, v in name_to_code_mapping.items()}

    inputs[0].set_data_from_numpy(image_data)
    results = triton_client.infer(model_name='classification_model',
                                  inputs=inputs,
                                  outputs=outputs)

    output0_data = results.as_numpy(output0_name)[0]
    output1_data = results.as_numpy(output0_name)[1]
    print(output0_data)
    print(code_to_name_mapping[output1_data])

    results = {
        'model_score' : str(output0_data),
        'predicted_class' : str(output1_data),
        'predicted_class_name' : code_to_name_mapping[output1_data]
    }

    with open("/workspace/model_repository/results.json", "w") as f:
        json.dump(results, f)

