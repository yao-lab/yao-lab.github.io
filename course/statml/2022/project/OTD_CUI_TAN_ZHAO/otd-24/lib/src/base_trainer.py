"""trainer code"""
import copy
import logging
import os
from typing import List, Dict, Optional, Callable, Union

import dill
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .logging_utils import loss_logger_helper

logger = logging.getLogger()


class Trainer:
    # This is like skorch but instead of callbacks we use class functions (looks less magic)
    # this is an evolving template
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim,
            scheduler: torch.optim.lr_scheduler,
            result_dir: Optional[str],
            statefile: Optional[str] = None,
            log_every: int = 100,
            save_strategy: Optional[List] = None,
            patience: int = 20,
            max_epoch: int = 100,
            stopping_criteria_direction: str = "bigger",
            stopping_criteria: Optional[Union[str, Callable]] = "accuracy",
            evaluations=None,
            **kwargs,
    ):
        """
            stopping_criteria : can be a function, string or none. If string it should match one
            of the keys in aux_loss or should be loss, if none we don't invoke early stopping
        """
        super().__init__()

        self.result_dir = result_dir
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluations = evaluations

        # training state related params
        self.epoch = 0
        self.step = 0
        self.best_criteria = None
        self.best_epoch = -1

        # config related param
        self.log_every = log_every
        self.save_strategy = save_strategy
        self.patience = patience
        self.max_epoch = max_epoch
        self.stopping_criteria_direction = stopping_criteria_direction
        self.stopping_criteria = stopping_criteria

        # TODO: should save config and see if things have changed?
        if statefile is not None:
            self.load(statefile)

        # init best model
        self.best_model = self.model.state_dict()

        # logging stuff
        if result_dir is not None:
            # we do not need to purge. Purging can delete the validation result
            self.summary_writer = SummaryWriter(log_dir=result_dir)

    def load(self, fname: str) -> Dict:
        """
            fname: file name to load data from
        """

        data = torch.load(open(fname, "rb"), pickle_module=dill, map_location=self.model.device)

        if hasattr(self, "model") and data.get("model") is not None:
            state_dict = self.model.state_dict()
            state_dict.update(data["model"])
            self.model.load_state_dict(state_dict)

        if hasattr(self, "optimizer") and data.get("optimizer") is not None:
            optimizer_dict = self.optimizer.state_dict()
            optimizer_dict.update(data["optimizer"])
            self.optimizer.load_state_dict(optimizer_dict)

        if hasattr(self, "scheduler") and data.get("scheduler") is not None:
            scheduler_dict = self.scheduler.state_dict()
            scheduler_dict.update(data["scheduler"])
            self.scheduler.load_state_dict(scheduler_dict)

        self.epoch = data["epoch"]
        self.step = data["step"]
        self.best_criteria = data["best_criteria"]
        self.best_epoch = data["best_epoch"]
        return data

    def save(self, fname: str, **kwargs):
        """
        fname: file name to save to
        kwargs: more arguments that we may want to save.

        By default we
            - save,
            - model,
            - optimizer,
            - epoch,
            - step,
            - best_criteria,
            - best_epoch
        """
        # NOTE: Best model is maintained but is saved automatically depending on save strategy,
        # So that It could be loaded outside of the training process
        kwargs.update({
                "model"        : self.model.state_dict(),
                "optimizer"    : self.optimizer.state_dict(),
                "epoch"        : self.epoch,
                "step"         : self.step,
                "best_criteria": self.best_criteria,
                "best_epoch"   : self.best_epoch,
        })

        if self.scheduler is not None:
            kwargs.update({"scheduler": self.scheduler.state_dict()})

        torch.save(kwargs, open(fname, "wb"), pickle_module=dill)

    def run_iteration(self, batch, training: bool = True, reduce: bool = True):
        """
            batch : batch of data, directly passed to model as is
            training: if training set to true else false
            reduce: whether to compute loss mean or return the raw vector form
        """
        pred = self.model(batch)
        loss, aux_loss = self.model.loss(pred, batch, reduce=reduce)

        if training:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss, aux_loss

    def compute_criteria(self, loss, aux_loss):
        stopping_criteria = self.stopping_criteria
        if stopping_criteria is None:
            return loss

        if callable(stopping_criteria):
            return stopping_criteria(loss, aux_loss)

        if stopping_criteria == "loss":
            return loss

        if aux_loss.get(stopping_criteria):
            return aux_loss[stopping_criteria]

        raise Exception(f"{stopping_criteria} not found")

    def train_batch(self, batch, *args, **kwargs):
        # This trains the batch
        loss, aux_loss = self.run_iteration(batch, training=True, reduce=True)
        loss_logger_helper(loss, aux_loss, writer=self.summary_writer, step=self.step,
                           epoch=self.epoch,
                           log_every=self.log_every, string="train")

    def train_epoch(self, train_loader, *args, **kwargs):
        # This trains the epoch and also calls on batch begin and on batch end
        # before and after calling train_batch respectively
        self.model.train()
        for i, batch in enumerate(train_loader):
            self.on_batch_begin(i, batch, *args, **kwargs)
            self.train_batch(batch, *args, **kwargs)
            self.on_batch_end(i, batch, *args, **kwargs)
            self.step += 1
        self.model.eval()

    def on_train_begin(self, train_loader, valid_loader, *args, **kwargs):
        # this could be used to add things to class object like scheduler etc
        if "init" in self.save_strategy:
            if self.epoch == 0:
                self.save(f"{self.result_dir}/init_model.pt")

    def on_epoch_begin(self, train_loader, valid_loader, *args, **kwargs):
        # This is called when epoch begins
        pass

    def on_batch_begin(self, epoch_step, batch, *args, **kwargs):
        # This is called when batch begins
        pass

    def on_train_end(self, train_loader, valid_loader, *args, **kwargs):
        # Called when training finishes. For base trainer we just save the last model
        if "last" in self.save_strategy:
            logger.info("Saving the last model")
            self.save(f"{self.result_dir}/last_model.pt")

    def on_epoch_end(self, train_loader, valid_loader, *args, **kwargs):
        # called when epoch ends
        # we call validation, scheduler here
        # also check if we have a new best model and save model if needed

        # call validate
        loss, aux_loss = self.validate(train_loader, valid_loader, *args, **kwargs)
        loss_logger_helper(loss, aux_loss, writer=self.summary_writer, step=self.step,
                           epoch=self.epoch, log_every=self.log_every, string="val",
                           force_print=True)

        # do scheduler step
        if self.scheduler is not None:
            prev_lr = [group['lr'] for group in self.optimizer.param_groups]
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                criteria = self.compute_criteria(loss, aux_loss)
                self.scheduler.step(criteria)
            else:
                self.scheduler.step()
            new_lr = [group['lr'] for group in self.optimizer.param_groups]

        # if you don't pass a criteria, it won't be computed and best model won't be saved.
        # on the contrary if you pass a stopping criteria, best model would be saved.
        # You can pass a large patience to get rid of early stopping
        if self.stopping_criteria is not None:
            criteria = self.compute_criteria(loss, aux_loss)

            if (
                    (self.best_criteria is None)
                    or (
                    self.stopping_criteria_direction == "bigger" and self.best_criteria < criteria)
                    or (
                    self.stopping_criteria_direction == "lower" and self.best_criteria > criteria)
            ):
                self.best_criteria = criteria
                self.best_epoch = self.epoch
                self.best_model = copy.deepcopy(
                    {k: v.cpu() for k, v in self.model.state_dict().items()})

                if "best" in self.save_strategy:
                    logger.info(f"Saving best model at epoch {self.epoch}")
                    self.save(f"{self.result_dir}/best_model.pt")

        if "epoch" in self.save_strategy:
            logger.info(f"Saving model at epoch {self.epoch}")
            self.save(f"{self.result_dir}/{self.epoch}_model.pt")

        if "current" in self.save_strategy:
            logger.info(f"Saving model at epoch {self.epoch}")
            self.save(f"{self.result_dir}/current_model.pt")

        # logic to load best model on reduce lr
        if self.scheduler is not None and not (all(a == b for (a, b) in zip(prev_lr, new_lr))):
            if getattr(self.scheduler, 'load_on_reduce', None) == "best":
                logger.info(f"Loading best model at epoch {self.epoch}")
                # we want to preserve the scheduler
                old_lrs = list(map(lambda x: x['lr'], self.optimizer.param_groups))
                old_scheduler_dict = copy.deepcopy(self.scheduler.state_dict())

                best_model_path = None
                if os.path.exists(f"{self.result_dir}/best_model.pt"):
                    best_model_path = f"{self.result_dir}/best_model.pt"
                else:
                    d = "/".join(self.result_dir.split("/")[:-1])
                    for directory in os.listdir(d):
                        if os.path.exists(f"{d}/{directory}/best_model.pt"):
                            best_model_path = self.load(f"{d}/{directory}/best_model.pt")

                if best_model_path is None:
                    raise FileNotFoundError(
                        f"Best Model not found in {self.result_dir}, please copy if it exists in "
                        f"other folder")

                self.load(best_model_path)
                # override scheduler to keep old one and also keep reduced learning rates
                self.scheduler.load_state_dict(old_scheduler_dict)
                for idx, lr in enumerate(old_lrs):
                    self.optimizer.param_groups[idx]['lr'] = lr
                logger.info(f"loaded best model and restarting from end of {self.epoch}")

    def on_batch_end(self, epoch_step, batch, *args, **kwargs):
        # called after a batch is trained
        pass

    def train(self, train_loader, valid_loader, *args, **kwargs):

        self.on_train_begin(train_loader, valid_loader, *args, **kwargs)
        while self.epoch < self.max_epoch:
            # NOTE: +1 here is more convenient, as now we don't need to do +1 before saving model
            # If we don't do +1 before saving model, we will have to redo the last epoch
            # So +1 here makes life easy, if we load model at end of e epoch, we will load model
            # and start with e+1... smooth
            self.epoch += 1
            self.on_epoch_begin(train_loader, valid_loader, *args, **kwargs)
            logger.info(f"Starting epoch {self.epoch}")
            self.train_epoch(train_loader, *args, **kwargs)
            self.on_epoch_end(train_loader, valid_loader, *args, **kwargs)

            if self.epoch - self.best_epoch > self.patience:
                logger.info(f"Patience reached stopping training after {self.epoch} epochs")
                break

        self.on_train_end(train_loader, valid_loader, *args, **kwargs)

    def validate(self, train_loader, valid_loader, *args, **kwargs):
        """
        we expect validate to return mean and other aux losses that we want to log
        """
        losses = []
        aux_losses = {}

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                loss, aux_loss = self.run_iteration(batch, training=False, reduce=False)
                losses.extend(loss.cpu().tolist())

                if i == 0:
                    for k, v in aux_loss.items():
                        # when we can't return sample wise statistics, we need to do this
                        if len(v.shape) == 0:
                            aux_losses[k] = [v.cpu().tolist()]
                        else:
                            aux_losses[k] = v.cpu().tolist()
                else:
                    for k, v in aux_loss.items():
                        if len(v.shape) == 0:
                            aux_losses[k].append(v.cpu().tolist())
                        else:
                            aux_losses[k].extend(v.cpu().tolist())
        return np.mean(losses), {k: np.mean(v) for (k, v) in aux_losses.items()}

    def test(self, train_loader, test_loader, *args, **kwargs):
        return self.validate(train_loader, test_loader, *args, **kwargs)
