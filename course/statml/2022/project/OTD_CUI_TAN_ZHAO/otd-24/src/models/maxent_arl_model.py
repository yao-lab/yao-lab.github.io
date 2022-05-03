import torch
import torch.nn.functional as F
from box import Box

from lib import BaseModel


class MaxentARL(BaseModel):
    """
        Implements MAXENT Adversarial model from Roy and Boddeti (2019)
    """

    def __init__(self, predictor, discriminator, encoder, z_size, beta, c_type, c_size, y_type,
                 y_size, *args, **kwargs):
        super().__init__()
        self._encoder = encoder
        self._predictor = predictor
        self._discriminator = discriminator

        self.y_type = y_type
        self.y_size = y_size
        self.c_size = c_size
        self.c_type = c_type

        self.device = "cpu"
        self.beta = beta  # adversary coefficient
        self.z_size = z_size

    def forward(self, batch):

        # move to device and compute forward
        x = batch[0].to(self.device)
        c = batch[1].to(self.device)
        y = batch[2].to(self.device)

        f = self.encode(x, c)
        y_hat = self.predict_logits(f)
        c_hat = self.predict_attr(f)

        return Box({"f": f, "y_hat": y_hat, "c_hat": c_hat})

    def encode(self, x, c):
        return self._encoder(x, c)

    def predict_logits(self, z):
        n = z.shape[0]

        if self.y_type == "binary":
            return self._predictor(z)
        if self.y_type == "one_hot":
            return F.log_softmax(self._predictor(z), dim=1)
        raise NotImplementedError()

    def predict_attr(self, z):
        n = z.shape[0]

        if self.c_type == "binary":
            return self._discriminator(z)
        if self.c_type == "one_hot":
            return F.log_softmax(self._discriminator(z), dim=1)
        raise NotImplementedError()

    def loss(self, pred, batch, reduce=True):
        x = batch[0].to(self.device)
        c = batch[1].to(self.device)
        y = batch[2].to(self.device)
        N = x.shape[0]  # batch size

        f = pred.f
        y_hat = pred.y_hat
        c_hat = pred.c_hat

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

        # get discriminator loss /entropy
        if self.c_type == "binary":
            disc_loss = F.binary_cross_entropy_with_logits(c_hat, c.float(),
                                                           reduction="none").squeeze()
            # 1/2 log p(y=1) + 1/2 log p(y=0)
            entropy = F.binary_cross_entropy_with_logits(c_hat, torch.ones_like(c.float()) / 2,
                                                         reduction="none").squeeze()
        elif self.c_type == "one_hot":
            # breakpoint()
            disc_loss = F.cross_entropy(c_hat, c.reshape(-1), reduction="none").squeeze()
            # taken from https://github.com/human-analysis/MaxEnt-ARL/blob/master/loss/entropy.py
            prob = torch.softmax(c_hat, dim=1)
            prob = prob + 1e-16  # for numerical stability while taking log
            entropy = -torch.sum(prob * torch.log(prob), dim=1)
        else:
            raise NotImplementedError()

        if reduce:
            pred_loss = pred_loss.mean()
            acc = acc.float().mean()
            disc_loss = disc_loss.mean()
            entropy = entropy.mean()

        # for loss we return pred_loss... but when do training we use beta*entropy + pred_loss
        # or disc_loss
        return pred_loss, {"pred_loss": pred_loss, "accuracy": acc, "disc_loss": disc_loss,
                           "entropy": entropy}
