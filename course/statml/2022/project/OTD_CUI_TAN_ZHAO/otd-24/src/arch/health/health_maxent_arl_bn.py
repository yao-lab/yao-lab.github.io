import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# should implement encoder and decoder (class, fn) which return a nn.Module
from box import Box


class Encoder(nn.Module):
    def __init__(self, input_size, z_size, c_size, c_type):
        super().__init__()
        if c_type == "binary":
            self.enc = nn.Sequential(
                nn.Linear(input_size[0] + math.ceil(math.log(c_size)), 100),
                nn.ReLU(),
                nn.Linear(100, z_size),
            )
        elif c_type == "one_hot":
            self.enc = nn.Sequential(
                nn.Linear(input_size[0] + c_size, 100),
                nn.ReLU(),
                nn.Linear(100, z_size),
            )
        self.c_type = c_type
        self.c_size = c_size

    def forward(self, x, c):
        n = x.shape[0]
        # c should be a integer here and we will one hot encode and pass it
        # breakpoint()
        if self.c_type == "one_hot":
            # if one_hot .. one hot encode it.
            # if binary we do nothing model will handle it
            c = F.one_hot(c, num_classes=self.c_size).float()
            c = c.squeeze()
        elif self.c_type == "binary":
            c = c.reshape(n, 1).float()
        else:
            # as is
            c = c.float()
        return self.enc(torch.cat([x, c], dim=1))


def predictor(input_size, z_size, y_size, y_type, c_size, c_type):
    if y_type == "binary":
        y_size = math.ceil(math.log(y_size))

    return nn.Sequential(nn.Linear(z_size, 50), nn.ReLU(), nn.Linear(50, y_size))


def discriminator(input_size, z_size, y_size, y_type, c_size, c_type):
    if c_type == "binary":
        c_size = math.ceil(math.log(c_size))

    return nn.Sequential(nn.BatchNorm1d(z_size), nn.Linear(z_size, 50), nn.BatchNorm1d(50),
                         nn.ReLU(), nn.Linear(50, c_size))


def get_arch(input_size, z_size, y_size, y_type, c_size, c_type, *args, **kwargs):
    return Box({"discriminator": discriminator(input_size, z_size, y_size, y_type, c_size, c_type),
                "predictor": predictor(input_size, z_size, y_size, y_type, c_size, c_type),
                "encoder": Encoder(input_size, z_size, c_size, c_type)})
