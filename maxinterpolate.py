import torch
import torch.nn as nn


class MaxInterpolate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, im0, im1):
        return torch.stack([im0, im1], 0).max(0)[0]
