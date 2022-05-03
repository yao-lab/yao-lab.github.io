import torch.nn.functional as F
from box import Box

from lib import BaseModel
from lib.src.samplers import sample_gaussian
from src.common.math import kl_gaussian


class Contrastive(BaseModel):
    """
        Base model to implement models with contrastive Estimation (InfoNCE)
    """

    def __init__(self, c_type, c_size, z_size, encoder, nce_estimator, latent_distribution, beta,
                 lambda_, *args, **kwargs):
        super().__init__()

        # this is extracted to have a handle to tune it from outside
        self.beta = beta
        self.lambda_ = lambda_

        self.c_type = c_type
        self.c_size = c_size
        self.z_size = z_size

        self._encoder = encoder
        self._nce_estimator = nce_estimator

        self.latent_distribution = latent_distribution

        self.device = "cpu"

    def to(self, device):
        self.device = device
        self._encoder.to(device)
        self._nce_estimator.to(device)
        return super().to(device=device)

    def _sample(self, f, s):
        if self.latent_distribution == "gaussian":
            z_sample = sample_gaussian(f, s)
            return z_sample

        raise NotImplementedError("Sampling method not implemented for latent distribution")

    def forward(self, batch):

        # move to device and compute forward
        x = batch[0].to(self.device)
        c = batch[1].to(self.device)

        f, s = self.encode(x)

        # sampling step
        z_sample = self._sample(f, s)

        return Box({"f": f, "s": s, "z_sample": z_sample})

    def encode(self, x):
        out = self._encoder(x)
        n = out.shape[0]

        if self.latent_distribution == "gaussian":
            # return mean and sigma
            out = out.view(n, -1, 2)
            mu = out[:, :, 0]
            sigma = F.softplus(out[:, :, 1])
            return mu, sigma

        raise NotImplementedError("latent distribution type is not implemented")

    def estimate_nce_f(self, x, c, z,y,idx):
        return self._nce_estimator(x, c, z,y,idx)

    def loss(self, pred, batch, reduce=True):

        # move to device and compute loss
        x = batch[0].to(self.device)
        c = batch[1].to(self.device)

        N = x.shape[0]  # batch size

        f, s = pred.f, pred.s
        z_sample = pred.z_sample

        # get KL
        if self.latent_distribution == "gaussian":
            kl = kl_gaussian(f, s)
        else:
            raise NotImplementedError()

        # get NCE
        mi = self.estimate_nce_f(x, c, z_sample)

        if reduce:
            kl = kl.mean()
            mi = mi.mean()

        return self.beta * kl + self.lambda_ * (-mi), {"kl": kl, "mi": mi}
