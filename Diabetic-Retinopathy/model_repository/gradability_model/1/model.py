import argparse
from lzma import MODE_NORMAL
import os
import sys
import time
from pprint import pprint
sys.path.append('/opt/tritonserver')
# print(sys.path)
from PIL import Image
import json
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from src.data.processing.augmentations import Augmentations
# import mlfoundry
# torch.set_default_dtype(torch.float32)

import src.utils.constants as constants
from src.main.helper import (
    cache_predictions,
    create_figures,
    epoch,
    get_dataloader,
    init_wandb,
    initialise_objs,
    load_checkpoints,
    load_ft_checkpoints,
    log_to_wandb,
    read_config,
    save_model_checkpoints,
    setup_checkpoint_dir,
    setup_misc_params,
    classify
)

MODEL_CONFIG_NAME = 'aiims_delhi/gradability_efficientnet_bce'
class ARGS:
    config_name = MODEL_CONFIG_NAME
MODEL_CONFIG_PATH = f'/opt/tritonserver/config/{MODEL_CONFIG_NAME}.yml'
class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        print("Initializing the model...")

        # client = mlfoundry.get_client()
        # model_version = client.get_model(os.getenv("GRADABILITY_MODEL_FQN")) # e.g. "model:truefoundry/user/iris-demo/iris-classifier:1"
        # self.model = model_version.load(map_location=torch.device('cpu'))
        
        # Define the relative path to your model file
        model_relative_path = "/model_repository/gradability_model/1/model.pth"
        

        # Get the current working directory (this will depend on where the script is being executed)
        current_dir = os.getcwd()

        # Combine the current directory with the relative path to get the absolute path
        model_path = os.path.join(current_dir, model_relative_path)
        print("model_path",model_path)

        # Check if the model file exists at the computed path
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at {model_path}")

        # Load the model from the computed path
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model_config = model_config = json.loads(args['model_config'])

        self.cfg = read_config(MODEL_CONFIG_PATH)

        setup_misc_params(self.cfg)


        # self.model, self.criterion, _, _ = initialise_objs(self.cfg, phase="inference")
        # self.model.to(self.device)
        # self.criterion.to(self.device)

        # ckpt_dir, root_dir = setup_checkpoint_dir(self.cfg, ARGS(), phase="inference")

        # print(f"Loading from checkpoints at {ckpt_dir}")

        # inference_checkpoint = self.cfg["inference"]["inference_checkpoint"]
        # self.model, _, _, _ = load_checkpoints(
        #     self.cfg, ARGS(), self.model, None, checkpoint_id=inference_checkpoint
        # )

        output0_config = pb_utils.get_output_config_by_name(model_config, "output_0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])

        # output1_config = pb_utils.get_output_config_by_name(model_config, "output_1")
        # self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config['data_type'])

        print('Initialized...')

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "raw_image")
            img = in_0.as_numpy()

            # print('krgndrkjfgndrskfn skfjnds klfs fkjs fksj fsjkf sljfsnefkjenbf')
            # print(img[0].shape)
            # print(np.squeeze(img).shape)
            # iitr
            img = img [0]

            transform = Augmentations(cfg = self.cfg['data']['augmentations'], use_augmentation = False,
                                    size = self.cfg['data']['imgsize'])

            # if transform is not None:
            transformed = transform(image=img)
            img = transformed["image"]
 
            img = img.to(self.device)

            imga = torch.unsqueeze(img, dim=0)

            self.model.eval()
            pred_scores = self.model(imga)

            pred_scores = pred_scores.cpu().detach()
            if (self.cfg["task_type"] == "binary-classification"):
                pred_scores = torch.sigmoid(pred_scores).numpy()
                # pred_scores = pred_scores.numpy()
                pred_labels = torch.zeros(pred_scores.shape).numpy()

                pred_labels[(pred_scores > self.cfg['inference']['threshold'])] = 1.0
                # print(pred_labels)
                # print(pred_scores)
                # print(pred_labels.shape)
                # print(pred_scores.shape)

            elif self.cfg["task_type"] == "multiclass-classification" and ("regression" in self.cfg["model"]["name"]):
                threshold = self.cfg['inference']['threshold']
                pred_scores = pred_scores.numpy()
                pred_labels = torch.tensor(
                        [classify(p, threshold) for p in pred_scores]
                    ).float()

                pred_labels = pred_labels.numpy()
            

            out_tensor_0 = pb_utils.Tensor("output_0",
                                           np.array([pred_scores[0],pred_labels[0]], dtype=object).astype(self.output0_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)  

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        # print(self.device)
        return responses


    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
