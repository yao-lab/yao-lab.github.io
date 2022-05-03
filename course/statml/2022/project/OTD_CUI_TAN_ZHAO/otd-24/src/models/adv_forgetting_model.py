import torch
import torch.nn.functional as F
from box import Box

from lib import BaseModel


class AdvForgetting(BaseModel):
    """
        Implements adv forgetting model from Jaiswal et.al 2020 (AAAI)
    """

    def __init__(self, encoder, predictor, discriminator, mask, decoder, c_type, y_type,
                 c_size, y_size, rho, delta, lambda_, z_size, *args, **kwargs):
        super().__init__()

        self.rho = rho
        self.delta = delta
        self.lambda_ = lambda_

        self.z_size = z_size

        self.c_type = c_type
        self.y_type = y_type
        self.c_size = c_size
        self.y_size = y_size

        self._encoder = encoder
        self._decoder = decoder
        self._predictor = predictor
        self._mask = mask
        self._discriminator = discriminator

        self.device = "cpu"
        self.counter = 0

    def forward(self, batch):

        # move to device and compute forward
        x = batch[0].to(self.device)
        c = batch[1].to(self.device)
        y = batch[1].to(self.device)

        z = self._encoder(x)
        x_hat = self._decoder(z)
        mask = self._mask(x)
        y_hat = self.predict_logits(z * mask)
        c_hat = self.predict_attr(z.detach() * mask)

        return Box({
            "z": z,
            "x_hat": x_hat,
            "y_hat": y_hat,
            "c_hat": c_hat,
            "mask": mask,
        })

    def predict_logits(self, z):
        if self.y_type == "binary":
            return self._predictor(z)
        if self.y_type == "one_hot":
            return F.log_softmax(self._predictor(z), dim=1)
        raise NotImplementedError()

    def predict_attr(self, z):
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

        z = pred.z
        mask = pred.mask
        x_hat = pred.x_hat
        y_hat = pred.y_hat
        c_hat = pred.c_hat

        distortion = F.mse_loss(x_hat.reshape(N, -1), x.reshape(N, -1), reduction="none").sum(dim=1)

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

        # get discriminator loss
        if self.c_type == "binary":
            disc_loss = F.binary_cross_entropy_with_logits(c_hat, c.float(),
                                                           reduction="none").squeeze()
        elif self.c_type == "one_hot":
            disc_loss = F.cross_entropy(c_hat, c.reshape(-1), reduction="none").squeeze()
        else:
            raise NotImplementedError()

        regularization = torch.sum(mask * (1 - mask), dim=1)

        # get acc
        # acc is an array here of 1,0 but after computing mean it will reflect acc
        acc = (pred_labels.squeeze() == y.squeeze().long()).long()

        if reduce:
            distortion = distortion.mean()
            pred_loss = pred_loss.mean()
            disc_loss = disc_loss.mean()
            acc = acc.float().mean()
            regularization = regularization.mean()

        return (
            pred_loss
            + self.rho * distortion
            + self.delta * disc_loss
            + self.lambda_ * regularization,
            {
                "distortion": distortion,
                "disc_loss": disc_loss,
                "pred_loss": pred_loss,
                "accuracy": acc,
                "regularization": regularization,
            },
        )
