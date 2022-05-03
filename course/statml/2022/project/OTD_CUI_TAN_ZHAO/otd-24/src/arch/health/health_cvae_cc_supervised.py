import math

import torch
import torch.nn as nn
from box import Box


def get_arch(input_size, z_size, y_size, y_type, c_size, c_type, *args, **kwargs):
    return Box({"encoder": encoder(input_size, z_size),
                "decoder": Decoder(input_size, z_size, c_size, c_type),
                "predictor": predictor(z_size, y_size, y_type, c_size, c_type),
                })


# should implement encoder and decoder (class, fn) which return a nn.Module
def encoder(input_size, z_size):
    return nn.Sequential(
        nn.Linear(input_size[0], 100),
        nn.ReLU(),
        nn.Linear(100, 2 * z_size),
    )


class Decoder(nn.Module):
    def __init__(self, inp_size, z, c_size=None, c_type="binary"):
        super().__init__()
        # if c_size is 2; we don't need on hot encoding, so we can just is

        if c_type == "binary":
            c_size = math.ceil(math.log2(c_size))

        # This just indicates that model is simply concatenating z.
        # This is useful when we want to visualize reconstruction
        # if C is not passed at all
        # i.e all zero C (makes not much sense .. but lets see)
        self.concat = True

        self.decoder = nn.Sequential(
            nn.Linear(z + c_size, 100),
            nn.ReLU(),
            nn.Linear(100, inp_size[0]),
        )

    def forward(self, z, c):
        z = torch.cat([z, c], dim=1)
        return self.decoder(z)


def predictor(z_size, y_size, y_type, c_size, c_type):
    if y_type == "binary":
        y_size = math.ceil(math.log(y_size))

    if c_type == "binary":
        c_size = math.ceil(math.log(c_size))

    return nn.Sequential(nn.Linear(z_size + c_size, 50), nn.ReLU(), nn.Linear(50, y_size))
