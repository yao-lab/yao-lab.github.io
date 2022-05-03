import torch
import torch.nn as nn
import torch.nn.functional as F
from box import Box

from lib import BaseModel
from lib.src.samplers import sample_gaussian
from src.common.math import kl_gaussian, conditional_marginal_divergence_gaussian


class CVIB(BaseModel):
    """
        Implements CVIB (Moyer et.al NeurIPS 2018)
        This is base class for CVIB model
    """

    def __init__(self, encoder, decoder, c_type, c_size, beta, lambda_, z_size, latent_distribution,
                 output_distribution, *args, **kwargs):
        super().__init__()

        # this is extracted to have a handle to tune it from outside
        self.beta = beta
        self.lambda_ = lambda_
        self.latent_distribution = latent_distribution
        self.output_distribution = output_distribution

        self.c_type = c_type
        self.c_size = c_size

        self.z_size = z_size

        self._encoder = encoder
        self._decoder = decoder

        self.device = "cpu"

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

        x_hat = self.decode(z_sample, c)
        return Box({"f": f, "s": s, "x_hat": x_hat})

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

    def decode(self, z, c):
        # if we have different decoder for each label
        if isinstance(self._decoder, nn.ModuleList):
            N = z.shape[0]
            c = c.cpu().long().view(-1)
            x_hat = [None] * N
            for c_ in set(c.tolist()):
                indices = torch.arange(N)[c == c_]
                temp = self._decoder[c_](z[indices])
                for i, idx in enumerate(indices):
                    x_hat[idx] = temp[i]
            x_hat = torch.stack(x_hat, dim=0)
        else:
            # c should be a integer here and we will one hot encode and pass it
            if self.c_type == "one_hot":
                # if one_hot .. one hot encode it.
                # if binary we do nothing model will handle it
                c = F.one_hot(c, num_classes=self.c_size)
                c = c.squeeze()
            x_hat = self._decoder(z, c.float())

        if self.output_distribution.lower() == "gaussian":
            return x_hat
        elif self.output_distribution.lower() == "bernoulli":
            return torch.sigmoid(x_hat)
        else:
            raise Exception(
                f"output distribution {self.output_distribution} not implemented for model")

    def loss(self, pred, batch, reduce=True):

        # move to device and compute loss
        x = batch[0].to(self.device)
        c = batch[1].to(self.device)
        N = x.shape[0]  # batch size

        f = pred.f
        s = pred.s
        x_hat = pred.x_hat

        # get distortion
        if self.output_distribution == "gaussian":
            distortion = F.mse_loss(x_hat.reshape(N, -1), x.reshape(N, -1), reduction="none").sum(
                dim=1)
        elif self.output_distribution == "bernoulli":
            distortion = F.binary_cross_entropy(x_hat.reshape(N, -1), x.reshape(N, -1),
                                                reduction="none").sum(dim=1)
        else:
            raise NotImplementedError()

        # get MI/KL
        if self.latent_distribution == "gaussian":
            kl = kl_gaussian(f, s)
        else:
            raise NotImplementedError()

        # get conditional-marginal-divergence
        if self.latent_distribution == "gaussian":
            cmd = conditional_marginal_divergence_gaussian(f, s)
        else:
            raise NotImplementedError()

        if reduce:
            distortion = distortion.mean()
            kl = kl.mean()
            cmd = cmd.mean()

        # kl can be approximated with cmd too.
        # and we use a beta-vae arch instead of vae from paper
        return (
            (1 + self.lambda_) * distortion
            + self.beta * kl
            + self.lambda_ * cmd,
            {"distortion": distortion, "kl": kl, "cmd": cmd},
        )
