"""List of supported custom collation functions for Pytorch dataloaders
"""

import numpy as np
import torch

def collate_fn_obj_detection(batch):
    tuple(zip(*batch))
