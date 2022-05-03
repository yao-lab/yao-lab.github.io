import math

import torch.nn as nn
from box import Box


def get_arch(input_size, z_size, y_size, y_type, c_size, c_type, *args, **kwargs):
    return Box({"encoder": encoder(input_size, z_size),
                "decoder": Decoder(input_size, z_size, c_size, c_type),
                "predictor": predictor(z_size, y_size, y_type),
                "mask": mask(input_size, z_size),
                "discriminator": discriminator(input_size, z_size, y_size, y_type, c_size, c_type)
                })


def mask(input_size, z_size):
    return nn.Sequential(
        nn.Linear(input_size[0], 100),
        nn.ReLU(),
        nn.Linear(100, z_size),
        nn.Sigmoid(),
    )


# should implement encoder and decoder (class, fn) which return a nn.Module
def encoder(input_size, z_size):
    return nn.Sequential(
        nn.Linear(input_size[0], 100),
        nn.ReLU(),
        nn.Linear(100, z_size),
    )


def discriminator(input_size, z_size, y_size, y_type, c_size, c_type):
    if c_type == "binary":
        c_size = math.ceil(math.log(c_size))

    return nn.Sequential(nn.Linear(z_size, 50), nn.ReLU(), nn.Linear(50, c_size))


class Decoder(nn.Module):
    def __init__(self, inp_size, z, c_size=None, c_type="binary"):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z, 100),
            nn.ReLU(),
            nn.Linear(100, inp_size[0]),
        )

    def forward(self, z):
        return self.decoder(z)


def predictor(z_size, y_size, y_type):
    if y_type == "binary":
        y_size = math.ceil(math.log(y_size))

    return nn.Sequential(nn.Linear(z_size, 50), nn.ReLU(), nn.Linear(50, y_size))
