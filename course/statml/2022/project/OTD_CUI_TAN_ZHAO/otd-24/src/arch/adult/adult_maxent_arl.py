import math

import torch
import torch.nn as nn
# should implement encoder and decoder (class, fn) which return a nn.Module
from box import Box


class Encoder(nn.Module):
    def __init__(self, input_size, z_size, c_size, c_type):
        super().__init__()
        if c_type == "binary":
            self.enc = nn.Sequential(
                nn.Linear(input_size[0] + math.ceil(math.log(c_size)), 50),
                nn.ReLU(),
                nn.Linear(50, z_size),
            )
        elif c_type == "one_hot":
            self.enc = nn.Sequential(
                nn.Linear(input_size[0] + c_size, 50),
                nn.ReLU(),
                nn.Linear(50, z_size),
            )

    def forward(self, x, c):
        return self.enc(torch.cat([x, c.reshape(-1, 1).float()], dim=1))


def predictor(input_size, z_size, y_size, y_type, c_size, c_type):
    if y_type == "binary":
        y_size = math.ceil(math.log(y_size))

    return nn.Sequential(nn.Linear(z_size, 50), nn.ReLU(), nn.Linear(50, y_size))


def discriminator(input_size, z_size, y_size, y_type, c_size, c_type):
    if c_type == "binary":
        c_size = math.ceil(math.log(c_size))

    return nn.Sequential(nn.Linear(z_size, 50), nn.ReLU(), nn.Linear(50, c_size))


def get_arch(input_size, z_size, y_size, y_type, c_size, c_type, *args, **kwargs):
    return Box({"discriminator": discriminator(input_size, z_size, y_size, y_type, c_size, c_type),
                "predictor": predictor(input_size, z_size, y_size, y_type, c_size, c_type),
                "encoder": Encoder(input_size, z_size, c_size, c_type)})
