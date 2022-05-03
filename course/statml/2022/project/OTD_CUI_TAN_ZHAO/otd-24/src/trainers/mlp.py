import logging

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from lib import BaseTrainer
from src.common import metrics

logger = logging.getLogger()


class MLPTrainer(BaseTrainer):
    def test(self, train_loader, test_loader, *args, **kwargs):
        loss, aux_loss = super().validate(train_loader, test_loader)
        # compute parity
        if self.model.y_type == "binary":
            y_hat, y, c = [], [], []
            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    y_logit = self.model.forward(batch)["y_hat"]
                    # y_hat is logit
                    y_hat.extend(torch.sigmoid(y_logit).cpu().tolist())
                    y.extend(batch[2].cpu().tolist())
                    c.extend(batch[1].cpu().tolist())

            y = np.array(y)
            c = np.array(c)
            # convert to numpy array and compute to probabilities
            prob = np.zeros((len(y_hat),2))
            prob[:, 0] = 1 - np.array(y_hat).reshape(-1)
            prob[:, 1] = np.array(y_hat).reshape(-1)
            # breakpoint()
            dp_soft, _ = metrics.demographic_parity_difference_soft(y, c, prob)
            dp, _ = metrics.demographic_parity_difference(y, c, prob)
            aux_loss.update({"demographic_parity_soft": dp_soft})
            aux_loss.update({"demographic_parity": dp})

            auc = roc_auc_score(y, prob[:, 1])
            aux_loss.update({"auc_score": auc})
        else:
            raise Exception("non-binary y_type is not handle. We need to fix this")
        return loss, aux_loss
