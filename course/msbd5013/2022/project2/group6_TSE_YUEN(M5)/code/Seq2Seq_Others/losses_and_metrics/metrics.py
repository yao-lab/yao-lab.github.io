import numpy as np


class WRMSSEMetric:
    def __init__(self):
        pass

    def get_error(self, yhat, y, scale, weight):
        error = (1/12) * np.sum(
            weight * np.sqrt(((y - yhat) ** 2).mean(1) / scale))
        return error


class SPLMetric:
    def __init__(self, quantiles=(0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995)):
        self.quantiles = np.array(quantiles)

    def get_error(self, yhat, y, scale, weight):
        yhat = yhat.transpose(2, 0, 1)  # Make quantile-major
        errors = y - yhat
        quantile_errors = errors * self.quantiles[:, None, None]
        quantile_errors[quantile_errors < 0] -= errors[quantile_errors < 0]
        error = (1/12) * np.sum(
            weight * quantile_errors.mean(2).mean(0) / scale)

        return error
