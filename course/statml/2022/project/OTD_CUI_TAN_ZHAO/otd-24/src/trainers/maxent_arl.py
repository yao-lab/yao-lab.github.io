import logging
import math
from typing import Dict, Optional, List, Union, Callable

import dill
import numpy as np
import torch
from box import Box
from sklearn.metrics import roc_auc_score

from lib import BaseTrainer
from src.common import metrics
from src.common.viz import do_classification

logger = logging.getLogger()


class MaxentARLTrainer(BaseTrainer):
    def __init__(self, model, optimizer: Dict, result_dir, statefile=None, log_every: int = 100,
                 save_strategy: Optional[List] = None, patience: int = 20, max_epoch: int = 100,
                 stopping_criteria_direction: str = "bigger",
                 stopping_criteria: Optional[Union[str, Callable]] = None,
                 evaluations: Optional[Dict] = None, **kwargs):
        super().__init__(model=model, optimizer=optimizer, result_dir=result_dir,
                         statefile=statefile, log_every=log_every, save_strategy=save_strategy,
                         patience=patience, max_epoch=max_epoch,
                         stopping_criteria_direction=stopping_criteria_direction,
                         stopping_criteria=stopping_criteria, evaluations=evaluations, **kwargs)
        self.model_update_freq = 2

    def load(self, fname):
        data = torch.load(open(fname, "rb"), pickle_module=dill, map_location=self.model.device)

        state_dict = self.model.state_dict()
        state_dict.update(data["model"])
        self.model.load_state_dict(state_dict)

        optimizer_dict = self.optimizer.disc.state_dict()
        optimizer_dict.update(data["optimizer_disc"])
        self.optimizer.disc.load_state_dict(optimizer_dict)

        optimizer_dict = self.optimizer.model.state_dict()
        optimizer_dict.update(data["optimizer_model"])
        self.optimizer.model.load_state_dict(optimizer_dict)

        self.epoch = data["epoch"]
        self.step = data["step"]
        self.best_criteria = data["best_criteria"]
        self.best_epoch = data["best_epoch"]
        return data

    def save(self, fname, **kwargs):
        kwargs.update({
            "model": self.model.state_dict(),
            "optimizer_disc": self.optimizer.disc.state_dict(),
            "optimizer_model": self.optimizer.model.state_dict(),
            "epoch": self.epoch,
            "step": self.step,
            "best_criteria": self.best_criteria,
            "best_epoch": self.best_epoch,
        })
        torch.save(kwargs, open(fname, "wb"), pickle_module=dill)

    def compute_encoding(self, loader):
        n = loader.dataset.tensors[0].shape[0]

        z = np.zeros((n, self.model.z_size))
        y = np.zeros(n)
        dim_c = math.ceil(math.log2(self.model.c_size)) if self.model.c_type == "binary" else 1

        c = np.zeros((n, dim_c))

        batch_size = loader.batch_size
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                x_ = batch[0].to(self.model.device)
                c_ = batch[1].to(self.model.device)

                temp = self.model.forward(batch)
                f = temp["f"]
                z[i * batch_size: (i + 1) * batch_size] = f.cpu().data.numpy()
                # batch has y labels
                if len(batch) > 2:
                    y[i * batch_size: (i + 1) * batch_size] = batch[2].view(-1).cpu().data.numpy()

                c[i * batch_size: (i + 1) * batch_size] = c_.cpu().data.numpy()

        result = Box({"z": z, "c": c})

        if len(batch) > 2:
            result["y"] = y

        return result

    def run_iteration(self, batch, training=True, reduce=True):
        update_model = self.step % self.model_update_freq == 0

        pred = self.model(batch)
        loss, aux_loss = self.model.loss(pred, batch, reduce=reduce)
        if training:
            if update_model:
                loss = aux_loss["pred_loss"] + self.model.beta * aux_loss["entropy"]
            else:
                loss = aux_loss["disc_loss"]

        if training:
            loss.backward()
            if update_model:
                self.optimizer.model.step()
            else:
                self.optimizer.disc.step()

            self.optimizer.disc.zero_grad()
            self.optimizer.model.zero_grad()

        return loss, aux_loss

    def validate(self, train_loader, valid_loader, *args, **kwargs):
        loss, aux_loss = super().validate(train_loader, valid_loader)
        return loss, aux_loss

    def test(self, train_loader, test_loader, *args, **kwargs):
        logger.info("Computing loss stats for test data")
        loss, aux_loss = super().validate(train_loader, test_loader)

        logger.info("Computing loss stats for train data")
        train_loss, train_aux_loss = super().validate(train_loader, train_loader)

        logger.info("Computing encodings for test data")
        temp = self.compute_encoding(test_loader)
        z_test, c_test, y_test = temp.z, temp.c, temp.y

        logger.info("Computing encodings for train data")
        temp = self.compute_encoding(train_loader)
        z_train, c_train, y_train = temp.z, temp.c, temp.y

        logger.info("Saving encoding")
        np.save(f"{self.result_dir}/embedding.npy", {"z_train": z_train,
                                                     "z_test": z_test,
                                                     "c_train": c_train,
                                                     "c_test": c_test,
                                                     "y_train": y_train,
                                                     "y_test": y_test,
                                                     })

        logger.info("Training a classifier on representation and compute accuracy")
        if y_train is not None and y_test is not None:
            score, rf_prob = do_classification(z_train, y_train, z_test, y_test,
                                               simple=True)
            aux_loss.update({"rf_acc": score})

            if self.model.y_type == "binary":
                # breakpoint()
                dp_rf, _ = metrics.demographic_parity_difference_soft(y_test, c_test, rf_prob)
                aux_loss.update({"demographic_parity_rf_soft": dp_rf})
                dp_rf, _ = metrics.demographic_parity_difference(y_test, c_test, rf_prob)
                aux_loss.update({"demographic_parity_rf": dp_rf})

                auc = roc_auc_score(y_test, rf_prob[:, 1])
                aux_loss.update({"rf_auc": auc})
            else:
                raise Exception("non-binary y_type is not handled")

            score, nn_prob = do_classification(z_train, y_train, z_test, y_test, simple=False)

            aux_loss.update({"nn_acc": score})
            if self.model.y_type == "binary":
                dp_nn, _ = metrics.demographic_parity_difference_soft(y_test, c_test, nn_prob)
                aux_loss.update({"demographic_parity_nn_soft": dp_nn})
                dp_nn, _ = metrics.demographic_parity_difference(y_test, c_test, nn_prob)
                aux_loss.update({"demographic_parity_nn": dp_nn})

                auc = roc_auc_score(y_test, nn_prob[:, 1])
                aux_loss.update({"nn_auc": auc})
            else:
                raise Exception("non-binary y_type is not handled")

        return loss, aux_loss
