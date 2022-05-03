import numpy as np
import torch
from scipy.linalg import circulant


def sample_gaussian(mean, sigma, tril_sigma=False):
    noise = torch.randn_like(mean)

    # we getting sigma
    if tril_sigma:
        z_sample = torch.bmm(sigma, noise.unsqueeze(dim=2)).squeeze() + mean
    else:
        z_sample = noise * sigma + mean
    return z_sample

