import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from box import Box


def get_arch(input_size, z_size, y_size, y_type, c_size, c_type, *args, **kwargs):
    return Box({"encoder": encoder(input_size, z_size),
                "nce_estimator": CPC(input_size, z_size, c_size),
                "predictor": predictor(z_size, y_size, y_type),
                })


# should implement encoder and decoder (class, fn) which return a nn.Module
def encoder(input_size, z_size):
    return nn.Sequential(
        nn.Linear(input_size[0], 50),
        nn.ReLU(),
        nn.Linear(50, 2 * z_size),
    )


class CPC(nn.Module):
    def __init__(self, input_size, z_size, c_size):
        super().__init__()

        self.f_x = nn.Sequential(
            nn.Linear(input_size[0], 50),
            nn.ReLU(),
            nn.Linear(50, z_size),
        )

        # just make one transform
        self.f_z = nn.Sequential(nn.Linear(z_size, z_size))
        self.w_s = nn.Parameter(data=torch.randn(c_size, z_size, z_size))
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return super().to(device=device)

    def forward(self, x, c, z):
        N = x.shape[0]
        f_x = self.f_x(x)
        f_z = self.f_z(z)

        temp = torch.bmm(torch.bmm(f_x.unsqueeze(2).transpose(1, 2), self.w_s[c.reshape(-1)]),
                         f_z.unsqueeze(2))
        T = F.softplus(temp.view(-1))

        neg_T = torch.zeros(N, device=self.device)

        for cat in set(c.reshape(-1).tolist()):
            f_z_given_c = f_z[(c == cat).reshape(-1)]
            f_x_given_c = f_x[(c == cat).reshape(-1)]

            # (N,Z) X (Z,Z)
            temp = F.softplus(f_x_given_c @ self.w_s[cat] @ f_z_given_c.transpose(0, 1))
            neg_T[(c == cat).reshape(-1)] = temp.mean(dim=1).view(-1)

        return torch.log(T + 1e-8) - torch.log(neg_T + 1e-8)


def predictor(z_size, y_size, y_type):
    if y_type == "binary":
        y_size = math.ceil(math.log(y_size))

    return nn.Sequential(nn.Linear(z_size, 50), nn.ReLU(), nn.Linear(50, y_size))
