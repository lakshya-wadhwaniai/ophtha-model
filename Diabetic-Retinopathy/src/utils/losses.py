"""All loss functions. Currently support losses from monai, torch and smp.
"""
# Read more here - https://smp.readthedocs.io/en/latest/losses.html
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (BCELoss, BCEWithLogitsLoss, CosineEmbeddingLoss,
                      CrossEntropyLoss, CTCLoss, GaussianNLLLoss,
                      HingeEmbeddingLoss, HuberLoss, KLDivLoss, L1Loss,
                      MarginRankingLoss, MSELoss, MultiLabelMarginLoss,
                      MultiLabelSoftMarginLoss, MultiMarginLoss, NLLLoss,
                      PoissonNLLLoss, SmoothL1Loss, SoftMarginLoss,
                      TripletMarginLoss, TripletMarginWithDistanceLoss)

import numpy as np

# from coral_pytorch.losses import (CoralLoss, CornLoss)
# Read more here - https://pytorch.org/docs/stable/nn.html#loss-functions

def one_hot(labels, num_classes, device, dtype):
    y = torch.eye(num_classes, device=device, dtype=dtype)
    return y[labels]

# class CoralLoss()/

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6

    def forward(self, input, target):
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)

def focal_loss(input, target, alpha, gamma=2.0, reduction='none', eps=1e-8):
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)

    alpha = alpha.to(input.device)

    # print(torch.log(input_soft).shape)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss