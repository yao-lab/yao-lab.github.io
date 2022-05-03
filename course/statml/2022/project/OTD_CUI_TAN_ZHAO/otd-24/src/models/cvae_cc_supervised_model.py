import torch
import torch.nn.functional as F
from box import Box

from .cvae_model import CVAE


class CVAEccSupervised(CVAE):
    """
        Implements CVAE with supervision and
        classification is conditioned on protected variable i.e. I(y:z|c)

        This is ablation of FCRL with reconstruction instead of InfoNCE loss
    """

    def __init__(self, c_type, c_size, y_type, y_size, predictor, encoder, decoder, z_size,
                 output_distribution, latent_distribution, beta, lambda_, *args,
                 **kwargs):
        super().__init__(encoder, decoder, c_type, c_size, z_size, output_distribution,
                         latent_distribution, beta, lambda_, *args, **kwargs)

        self.y_size = y_size
        self.y_type = y_type
        self._predictor = predictor

        self.device = "cpu"
        self.lambda_ = lambda_

    def forward(self, batch):

        # move to device and compute forward
        x = batch[0].to(self.device)
        c = batch[1].to(self.device)

        f, s = self.encode(x)

        # sampling step
        z_sample = self._sample(f, s)

        x_hat = self.decode(z_sample, c)

        y_hat = self.predict_logits(z_sample, c)

        return Box({"f": f, "s": s, "x_hat": x_hat, "y_hat": y_hat})

    def predict_logits(self, z, c):
        n = z.shape[0]
        # c should be a integer here and we will one hot encode and pass it
        if self.c_type == "one_hot":
            # if one_hot .. one hot encode it.
            # if binary we do nothing model will handle it
            c = F.one_hot(c, num_classes=self.c_size).float()
            c = c.squeeze()
        elif self.c_type == "binary":
            c = c.reshape(n, 1).float()
        else:
            # as is
            c = c.float()

        if self.y_type == "binary":
            return self._predictor(torch.cat([z, c], dim=1))
        if self.y_type == "one_hot":
            return F.log_softmax(self._predictor(torch.cat([z, c], dim=1)), dim=1)
        raise NotImplementedError()

    def loss(self, pred, batch, reduce=True):

        x = batch[0].to(self.device)
        c = batch[1].to(self.device)
        y = batch[2].to(self.device)
        N = x.shape[0]  # batch size

        _, temp = super().loss(pred, [x, c], reduce=False)
        distortion = temp["distortion"]
        kl = temp["kl"]

        f = pred.f
        s = pred.s
        x_hat = pred.x_hat
        y_hat = pred.y_hat

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
            distortion = distortion.mean()
            kl = kl.mean()
            pred_loss = pred_loss.mean()
            acc = acc.float().mean()

        return (pred_loss + self.lambda_ * distortion + self.beta * kl, {
            "distortion": distortion, "kl": kl, "pred_loss": pred_loss, "accuracy": acc,
        })
