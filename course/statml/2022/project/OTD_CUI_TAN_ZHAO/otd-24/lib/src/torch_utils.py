""" things torch should have but it doesn't"""
import logging

import torch
import torch.nn as nn
from torch.autograd import Function

logger = logging.getLogger()
EPSILON = 1e-8


# reset seed
def reset_seed():
    while True:
        try:
            torch.seed()
        except RuntimeError as _:
            logger.error("Error generating seed")
        else:
            break


class Reshape(nn.Module):
    """
        Reshape module that reshapes any input to (batch_size, ...shape)
        by default it does flattening but you can pass any shape.
    """

    def __init__(self, shape=(-1,)):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view((batch_size,) + self.shape)

    def extra_repr(self):
        return f"shape={self.shape}"


class Offset(torch.nn.Module):
    def __init__(self, offset, net):
        super().__init__()
        self.offset = nn.Parameter(offset, requires_grad=False)
        self.net = net

    def forward(self, *args):
        batch_size = args[0].shape[0]
        return self.offset.expand((batch_size, -1, -1, -1)) + 1e-8  # + self.net(*args)


def batch_eye(N, D, device="cpu"):
    x = torch.eye(D, device=device)
    x = x.unsqueeze(0)
    x = x.repeat(N, 1, 1)
    return x


def batch_eye_like(tensor):
    assert len(tensor.shape) == 3 and tensor.shape[1] == tensor.shape[2]
    N = tensor.shape[0]
    D = tensor.shape[1]
    return batch_eye(N, D, device=tensor.device)

class _RevGrad(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input


revgrad = _RevGrad.apply


class RevGrad(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

    def forward(self, input_):
        return revgrad(input_)
