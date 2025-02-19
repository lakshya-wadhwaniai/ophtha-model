from os.path import join
from src.data.datasets.base_dr_dataset import Base_DR_Dataset

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
# import torch.nn.functional
from torch.utils.data import Dataset

from src.utils.constants import EYEPACS_ROOT_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True

# mean ([0.3233, 0.2271, 0.1634])
# std  ([0.2635, 0.1848, 0.1334])


class EYEPACS(Base_DR_Dataset):
    def __init__(
        self,
        ROOT = EYEPACS_ROOT_DIR,
        transform = None,
        phase = "train",
        fraction = 1.0,
        upsample = False,
        upsample_labels = [],
        upsample_factors = [],
        split_version = "v1_cv2_1024_0.8_0.1_0.1",
        Gfilter = False,
        **kwargs,
    ):

        super().__init__(
            ROOT=ROOT,
            split_version=split_version,
            transform=transform,
            phase=phase,
            fraction=fraction,
            upsample=upsample,
            upsample_labels=upsample_labels,
            upsample_factors=upsample_factors,
            Gfilter = Gfilter,
            **kwargs
            )


    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_path = self.images_list.iloc[idx]['image']
        image = self.processImage(image_path)
        label = self.images_list.iloc[idx]['level']
        label_vector = torch.tensor(label)
        return (image , label_vector, image_path)

if __name__ == "__main__":
    dataset = EYEPACS()
    # mean = 0.0
    # std = 0.0
    # arr = []
    # for img, _, l in dataset:
    #     arr.append(img)
    #     #mean += img.sum([1,2])/torch.numel(img[0])
    #     mean += img.mean(2)
    #     std += img.std(2)

    # mean /= len(dataset)
    # std /= len(dataset)
    # print(mean)
    # print(std)
    # print(arr.mean(2))
    # print(arr.std(2))
