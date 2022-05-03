import torch.nn as nn
from box import Box


def get_arch(input_size, z_size, y_size, y_type, c_size, c_type, *args, **kwargs):
    return Box({"predictor": predictor(input_size, y_size, y_type)})


def predictor(input_size, y_size, y_type):
    return nn.Sequential(
        nn.Linear(input_size[0], 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )
