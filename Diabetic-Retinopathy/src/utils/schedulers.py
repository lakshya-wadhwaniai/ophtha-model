"""All Learning Rate Schedulers. Currently only support torch LR Schedulers.
"""
import torch
from torch.optim.lr_scheduler import (ChainedScheduler, ConstantLR,
                                      CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, CyclicLR,
                                      ExponentialLR, LambdaLR, LinearLR,
                                      MultiplicativeLR, MultiStepLR,
                                      OneCycleLR, ReduceLROnPlateau,
                                      SequentialLR, StepLR)


class ClippedCosineAnnealingLR():
    def __init__(self, optimizer, T_max, min_lr):
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        self.min_lr = min_lr
        self.finish = False

    def step(self):
        if not self.finish:
            self.scheduler.step()
            curr_lr = self.optimizer.param_groups[0]['lr']
            if curr_lr < self.min_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.min_lr
                self.finish = True

    def is_finish(self):
        return self.finish