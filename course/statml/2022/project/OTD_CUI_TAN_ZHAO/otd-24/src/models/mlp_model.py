import torch
import torch.nn.functional as F
from box import Box

from lib import BaseModel


class MLP(BaseModel):
    """
        Implements MLP
    """

    def __init__(self, predictor, y_type, *args, **kwargs):
        super().__init__()
        self.y_type = y_type

        self._predictor = predictor
        self.device = "cpu"

    def forward(self, batch):

        # move to device and compute forward
        x = batch[0].to(self.device)
        y_hat = self.predict(x)
        return Box({"y_hat": y_hat})

    def predict(self, x):
        """Output logits"""
        y_hat = self._predictor(x)
        if self.y_type == "binary":
            return y_hat
        elif self.y_type == "one_hot":
            return torch.log_softmax(y_hat, dim=1)
        else:
            raise Exception(f"{self.y_type} type not found for y")

    def loss(self, pred, batch, reduce=True):

        # move to device and compute loss
        x = batch[0].to(self.device)
        y = batch[2].to(self.device)
        N = x.shape[0]  # batch size

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

        acc = (pred_labels.squeeze() == y.squeeze().long()).float()
        if reduce:
            pred_loss = pred_loss.mean()
            acc = acc.mean()

        return pred_loss, {"pred_loss": pred_loss, "accuracy": acc}
