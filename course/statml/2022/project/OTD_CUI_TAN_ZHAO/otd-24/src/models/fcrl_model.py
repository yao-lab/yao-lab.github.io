import torch
import torch.nn.functional as F
from box import Box
import numpy as np
from random import shuffle
from src.common.math import kl_gaussian
from .contrastive_model import Contrastive
from sklearn.metrics import f1_score


class FCRL(Contrastive):
    """
        Implements Invariance model with InfoNCE and classification loss conditioned on c
    """

    def __init__(self, c_type, c_size, c_mix_num,y_type, y_size, z_size, predictor, encoder, nce_estimator,
                 latent_distribution, beta, lambda_, *args, **kwargs):
        super().__init__(c_type, c_size, z_size, encoder, nce_estimator, latent_distribution, beta,
                         lambda_, *args, **kwargs)

        # this is extracted to have a handle to tune it from outside
        self.beta = beta
        self.lambda_ = lambda_
        self.c_mix_num=c_mix_num

        self.y_size = y_size
        self.y_type = y_type

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

        y_hat = self.predict_logits(z_sample, c)

        return Box({"f": f, "s": s, "z_sample": z_sample, "y_hat": y_hat})

    def predict_logits(self, z, c):
        n = z.shape[0]
        # c should be a integer here and we will one hot encode and pass it

        if type(self.c_size)!=int:

            if self.c_size[0]==1:
                tmp=c[:,0].reshape(n, 1).float()
            else:
                tmp = F.one_hot(c[:,0], num_classes=self.c_size[0]).float()
            for i in range(1,len(self.c_size)):
                if self.c_size[i]==1:
                    tmp = torch.cat((tmp,c[:,i].reshape(n, 1).float()),dim=1)
                else:
                    tmp = torch.cat((tmp,F.one_hot(c[:,i], num_classes=self.c_size[i]).float()),dim=1)

            c=tmp
            c = c.squeeze()
            # tmp = F.one_hot(c[:,0], num_classes=self.c_size[0]).float()
            # for i in range(1,len(self.c_size)):
            #
            #     tmp = torch.cat((tmp,F.one_hot(c[:,i], num_classes=self.c_size[i]).float()),dim=1)
            # c=tmp
            # c = c.squeeze()

        elif self.c_type == "one_hot":
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
            #return self._predictor(torch.cat([z, c], dim=1))
            return self._predictor(z)

        if self.y_type == "one_hot":
            return F.log_softmax(self._predictor(torch.cat([z, c], dim=1)), dim=1)
        raise NotImplementedError()

    def loss(self, pred, batch, reduce=True):

        # move to device and compute loss
        x = batch[0].to(self.device)
        c = batch[1].to(self.device)
        y = batch[2].to(self.device)
        N = x.shape[0]  # batch size

        f, s, y_hat = pred.f, pred.s, pred.y_hat
        z_sample = pred.z_sample

        n = c.shape[0]
        # c should be a integer here and we will one hot encode and pass it
        if type(self.c_size)!=int:
            if self.c_size[0]==1:
                tmp=c[:,0].reshape(n, 1).float()
            else:
                tmp = F.one_hot(c[:,0], num_classes=self.c_size[0]).float()
            for i in range(1,len(self.c_size)):
                if self.c_size[i]==1:
                    tmp = torch.cat((tmp,c[:,i].reshape(n, 1).float()),dim=1)
                else:
                    tmp = torch.cat((tmp,F.one_hot(c[:,i], num_classes=self.c_size[i]).float()),dim=1)

            c=tmp
            c = c.squeeze()

        elif self.c_type == "one_hot":
            # if one_hot .. one hot encode it.
            # if binary we do nothing model will handle it
            c = F.one_hot(c, num_classes=self.c_size).float()
            c = c.squeeze()
        elif self.c_type == "binary":
            c = c.reshape(n, 1).float()
        else:
            # as is
            c = c.float()

        # get MI/KL
        if self.latent_distribution == "gaussian":
            kl = kl_gaussian(f, s)
        else:
            raise NotImplementedError()

        # get NCE

        N = x.shape[0]
        #split
        idx=list(np.arange(N))
        shuffle(idx)
        mi = self.estimate_nce_f(x, c, z_sample,y,idx)

        label_mi=torch.ones_like(mi)
        label_mi[idx[int(N/2):]]=0.0
        mi_loss = F.binary_cross_entropy_with_logits(mi, label_mi, reduction="none").squeeze()

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
            mi_loss=mi_loss.mean()
            pred_loss = pred_loss.mean()
            acc = acc.float().mean()

        return pred_loss + self.beta * kl + self.lambda_ * (mi_loss), {
            "kl": kl, "mi": mi_loss, "accuracy": acc, "pred_loss": pred_loss
        }
class FCRL_old(Contrastive):
    """
        Implements Invariance model with InfoNCE and classification loss conditioned on c
    """

    def __init__(self, c_type, c_size, y_type, y_size, z_size, predictor, encoder, nce_estimator,
                 latent_distribution, beta, lambda_, *args, **kwargs):
        super().__init__(c_type, c_size, z_size, encoder, nce_estimator, latent_distribution, beta,
                         lambda_, *args, **kwargs)

        # this is extracted to have a handle to tune it from outside
        self.beta = beta
        self.lambda_ = lambda_

        self.y_size = y_size
        self.y_type = y_type

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

        y_hat = self.predict_logits(z_sample, c)

        return Box({"f": f, "s": s, "z_sample": z_sample, "y_hat": y_hat})

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

        return pred_loss + self.beta * kl + self.lambda_ * (-mi), {
            "kl": kl, "mi": mi, "accuracy": acc, "pred_loss": pred_loss
        }
