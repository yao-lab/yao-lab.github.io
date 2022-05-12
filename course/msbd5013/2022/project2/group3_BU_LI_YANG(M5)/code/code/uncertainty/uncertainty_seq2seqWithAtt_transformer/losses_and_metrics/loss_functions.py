import torch
import torch.nn as nn
import numpy as np


class MSELoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, yhat, y, scale, weight):
        loss = torch.mean(
            self.mse(yhat, y).mean(1))
        return loss


class RMSELoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, yhat, y, scale, weight):
        loss = torch.mean(
            torch.sqrt(self.mse(yhat, y).mean(1)))
        return loss


class RMSSELoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, yhat, y, scale, weight):
        loss = torch.mean(
            torch.sqrt(self.mse(yhat, y).mean(1) / scale))
        return loss


class WRMSSELevel12Loss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, yhat, y, scale, weight):
        loss = torch.sum(
            weight * torch.sqrt(self.mse(yhat, y).mean(1) / scale))
        return loss


class WRMSSELoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, yhat, y, scale, weight):
        loss = (1/12) * torch.sum(
            weight * torch.sqrt(self.mse(yhat, y).mean(1) / scale))
        return loss


class SPLLevel12Loss(nn.Module):
    def __init__(self, config, quantiles=np.array([0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995])):
        super().__init__()
        self.quantiles = torch.from_numpy(quantiles).to(config.device)

    def forward(self, yhat, y, scale, weight):
        yhat = yhat.permute(2, 0, 1)  # Make quantile-major
        errors = y - yhat
        quantile_errors = errors * self.quantiles[:, None, None]
        quantile_errors[quantile_errors < 0] -= errors[quantile_errors < 0]
        loss = torch.sum(
            weight * quantile_errors.mean(2).mean(0) / scale)

        return loss


class SPLLoss(nn.Module):
    def __init__(self, config, quantiles=np.array([0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995])):
        super().__init__()
        self.quantiles = torch.from_numpy(quantiles).to(config.device)

    def forward(self, yhat, y, scale, weight):
        yhat = yhat.permute(2, 0, 1)  # Make quantile-major
        errors = y - yhat
        quantile_errors = errors * self.quantiles[:, None, None]
        quantile_errors[quantile_errors < 0] -= errors[quantile_errors < 0]
        loss = (1/12) * torch.sum(
            weight * quantile_errors.mean(2).mean(0) / scale)

        return loss
