import torch
import torch.nn.functional as F
from box import Box

from src.common.math import kl_gaussian
from .contrastive_model import Contrastive


class FCRL_NO_Conditioning(Contrastive):
    """
        Implements Invariance model with NCE and classification loss
    """

    def __init__(self, c_type, c_size, y_type, y_size, z_size, predictor, encoder, nce_estimator,
                 latent_distribution, beta, lambda_, *args, **kwargs):
        super().__init__(c_type, c_size, z_size, encoder, nce_estimator, latent_distribution, beta,
                         lambda_, *args, **kwargs)

        # this is extracted to have a handle to tune it from outside
        self.y_type = y_type
        self.y_size = y_size

        self._predictor = predictor

        self.device = "cpu"

    def to(self, device):
        self.device = device
        self._predictor.to(device)
        return super().to(device=device)

    def forward(self, batch):

        # move to device and compute forward
        x = batch[0].to(self.device)
        c = batch[1].to(self.device)

        f, s = self.encode(x)

        # sampling step
        z_sample = self._sample(f, s)

        y_hat = self.predict_logits(z_sample)

        return Box({"f": f, "s": s, "z_sample": z_sample, "y_hat": y_hat})

    def predict_logits(self, z):
        n = z.shape[0]
        if self.y_type == "binary":
            return self._predictor(z)
        if self.y_type == "one_hot":
            return F.log_softmax(self._predictor(z))
        raise NotImplementedError()

    def loss(self, pred, batch, reduce=True):

        # move to device and compute loss
        x = batch[0].to(self.device)
        c = batch[1].to(self.device)
        y = batch[2].to(self.device)
        N = x.shape[0]  # batch size

        f, s, y_hat = pred.f, pred.s, pred.y_hat
        z_sample = pred.z_sample

        # get MI/KL
        if self.latent_distribution == "gaussian":
            kl = kl_gaussian(f, s)
        else:
            raise NotImplementedError()

        # get NCE
        mi = self.estimate_nce_f(x, c, z_sample)

        # get prediction loss
        if self.y_type == "binary":
            pred_loss = F.binary_cross_entropy_with_logits(y_hat, y.float(),
                                                           reduction="none").squeeze()
            pred_labels = (y_hat > 0).long()
        elif self.y_type == "one_hot":
            pred_loss = F.cross_entropy(y_hat, y, reduction="none").squeeze()
            pred_labels = torch.argmax(y_hat, dim=1)
        else:
            raise NotImplementedError()

        # get acc
        # acc is an array here of 1,0 but after computing mean it will reflect acc
        acc = (pred_labels.squeeze() == y.squeeze().long()).long()

        if reduce:
            kl = kl.mean()
            mi = mi.mean()
            pred_loss = pred_loss.mean()
            acc = acc.float().mean()

        return pred_loss + self.beta * kl + self.lambda_ * (-mi), {"kl": kl, "mi": mi,
                                                                   "accuracy": acc,
                                                                   "pred_loss": pred_loss}
