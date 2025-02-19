[![py39](https://img.shields.io/badge/python-3.9-brightgreen.svg)](https://img.shields.io/badge/python-3.9-brightgreen.svg)
[![cu117](https://img.shields.io/badge/cuda-11.7-blue.svg)](https://img.shields.io/badge/cuda-11.7-blue.svg)

# Diabetic Retinopathy Classification
Diabetic retinopathy (DR) is a complication that affects the eyes in diabetics. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina).
The scope of this repo is to create an image classification model to predict the different grades of DR from [No DR, Mild, Moderate, Severe, Proliferative] from the retinal fundus images. Additionally we can train a binary Referable/Non Referable model and a DR/No DR model.

## Getting started
- Clone the repo
```sh
$ git clone https://github.com/WadhwaniAI/Diabetic-Retinopathy.git
$ cd Diabetic-Retinopathy
```

- Create a conda environment and install dependencies
```sh
$ conda create --name venv
$ conda activate venv
$ pip install -r requirements.txt
```

## Repository overview
- `configs/`: contain configs and hyperparameters for training a model.
- `src/models/`: class definitions for the different models used
- `src/data/`: contains code for loading data, and doing various augmentations, including one time data preprocessing.
- `src/utils/`: contains scripts for helper functionalities like metrics, losses, constants
- `src/main/`: scripts for training, evaluation, and other functions related to training

## Preparing the Data
### Downloading the Data
- Download the Eyepacs data from [Kaggle DR Challenge](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data)
- Aptos Dataset from [APTOS Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
- ODIR Dataset from [Ocular Disease Recognition](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) (We only use the DR and Normal Fundus images from this dataset)

### Cropping images
After Downloading, run the [cropping script](src/data/processing/crop_images.py) for each dataset, to crop all images to a uniform 1024x1024 resolution.

### Creating the splits
The [splits script](src/data/processing/create_splits.py) will read the annotation file of the data and create train, val, and test splits in a .csv format expected by the dataset file. 

Extend the base [dataset class](src/data/datasets/base_dr_dataset.py) to run a custom dataset.

## Train the model
```sh
$PYTHONPATH='/.../Diabetic-Retinopathy' \
$python src/main/train.py --config multiclass_model.yml
```

## Run evaluation 
- on the validation set
```sh
$PYTHONPATH='/.../Diabetic-Retinopathy' \
$python src/main/eval.py --config multiclass_model.yml --eval_mode val
```

- on the test set
```sh
$PYTHONPATH='/.../Diabetic-Retinopathy' \
$python src/main/eval.py --config multiclass_model.yml --eval_mode test
```

## ⚠️ **NOTE**
- Training the model on an existing version overwrites the existing checkpoints. For reproducibility, always copy and create a new yaml in `configs/` with the next available version name and then run training with the new version
- On-prem data path: `/scratchk/ehealth/optha/eyepacs/data/`


## Author
- [Rishabh Sharma](https://github.com/rishabh-WIAI)

## Maintainers
- [Rishabh Sharma](https://github.com/rishabh-WIAI)
- [Mukul](https://github.com/mukul-wai)
