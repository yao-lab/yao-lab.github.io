import torch
import torch.nn.functional as F
from box import Box

from .cvib_model import CVIB


class CVIBSupervised(CVIB):
    """ extend CVIB for supervised bottleneck"""

    def __init__(self, predictor, encoder, decoder, y_type, y_size, c_type, c_size, beta, lambda_,
                 z_size, use_posterior_sample, latent_distribution, output_distribution,
                 *args, **kwargs):
        super().__init__(encoder, decoder, c_type, c_size, beta, lambda_, z_size,
                         latent_distribution, output_distribution, *args, **kwargs)

        self.y_type = y_type
        self.y_size = y_size

        self._predictor = predictor

        self.use_posterior_sample = use_posterior_sample

        self.device = "cpu"

    def forward(self, batch):

        # move to device and compute forward
        x = batch[0].to(self.device)
        c = batch[1].to(self.device)

        f, s = self.encode(x)

        # sampling step
        z_sample = self._sample(f, s)

        x_hat = self.decode(z_sample, c)

        if self.use_posterior_sample:
            y_hat = self.predict_logits(z_sample)
        else:
            y_hat = self.predict_logits(f)

        return Box({"f": f, "s": s, "x_hat": x_hat, "y_hat": y_hat})

    def predict_logits(self, z):
        if self.y_type == "binary":
            return self._predictor(z)
        if self.y_type == "one_hot":
            return F.log_softmax(self._predictor(z), dim=1)
        raise NotImplementedError()

    def loss(self, pred, batch, reduce=True):

        x = batch[0].to(self.device)
        c = batch[1].to(self.device)
        y = batch[2].to(self.device)
        N = x.shape[0]  # batch size

        _, temp = super().loss(pred, [x, c], reduce=False)
        distortion = temp["distortion"]
        kl = temp["kl"]
        cmd = temp["cmd"]

        f = pred.f
        s = pred.s
        x_hat = pred.x_hat
        y_hat = pred.y_hat

        # get prediction loss

        if self.y_type == "binary":
            pred_loss = F.binary_cross_entropy_with_logits(
                y_hat, y.float(), reduction="none").squeeze()
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
            distortion = distortion.mean()
            kl = kl.mean()
            pred_loss = pred_loss.mean()
            cmd = cmd.mean()
            acc = acc.float().mean()

        return pred_loss + self.lambda_ * (distortion + cmd) + (self.beta * cmd), {
            "distortion": distortion,
            "kl": kl,
            "pred_loss": pred_loss,
            "accuracy": acc,
            "cmd": cmd,
        },
