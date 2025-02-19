import os
import random
from os.path import join

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Base_DR_Dataset(Dataset):
    def __init__(
        self,
        ROOT,
        split_version,
        transform,
        phase,
        fraction,
        upsample = False,
        upsample_labels = [],
        upsample_factors = [],
        Gfilter = False,
        **kwargs,
    ):
        self.class_names = {
            "No DR": 0,
            "Mild": 1,
            "Moderate": 2,
            "Severe": 3,
            "Proliferative": 4,
        }
        self.ROOT = ROOT
        self.transform = transform
        self.splits_dir = join(ROOT, "splits")
        self.Gfilter = Gfilter

        self.splitfile_path = os.path.join(
            self.splits_dir, split_version ,f"{phase}.csv"
        )

        self.images_list = self.get_image_list()

        if upsample and phase == "train":
            upsample_image_list = self.upsample_positives(
                upsample_labels, upsample_factors
            )
            self.images.extend(upsample_image_list)

    def get_image_list(self):
        """
            Read data from splits file and shuffle
        """

        images = pd.read_csv(self.splitfile_path)

        print(f"> Total data samples found : {len(images)}")

        return images

    def processImage(self, image_name):
        """
        Load and apply transformation to image.
        """
        temp = Image.open(image_name).convert("RGB")
        image = np.asarray(temp)

        if self.Gfilter:
            image = cv2.addWeighted(image,4, cv2.GaussianBlur( image , (0,0) , self.Gfilter) ,-4 ,128)
            
        temp.close()

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image

    def upsample_positives(self, upsample_labels_list, upsample_factor):
        upsampled_image_list = []
        for it, label in enumerate(upsample_labels_list):
            label_images = self.label2image_dict[label]
            label_images = [x for x in label_images if x in self.images_list]
            # Subtracting one because we need to calculate "increase" in samples
            upsample_len = len(label_images) * (upsample_factor[it] - 1)
            upsampled_label_images = random.choices(label_images, k=upsample_len)

            upsampled_image_list.extend(upsampled_label_images)

        return upsampled_image_list


    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_path = self.images_list.iloc[idx]['image']
        image_path = join(self.ROOT, image_path.split('/')[-2], image_path.split('/')[-1])
        image = self.processImage(image_path)
        label_vector = self.images_list.iloc[idx]['level']
        label_vector = torch.from_numpy(label_vector)
        return (image , label_vector, image_path)
  
if __name__ == "__main__":
    dataset = Base_DR_Dataset()
