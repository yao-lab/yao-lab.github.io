import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from importlib import import_module
import glob
import os
import sys
import shutil

from data_generator import DataLoader
from utils.data_utils import *
from utils.training_utils import ModelCheckpoint, EarlyStopping
from losses_and_metrics import loss_functions, metrics

class Trainer:
    def __init__(self, config):
        self.config = config
        self.terminal_width = shutil.get_terminal_size((80, 20)).columns

        # Model
        print(f' Model: {self.config.architecture} '.center(self.terminal_width, '*'))
        model_type = import_module('models.' + self.config.architecture)
        create_model = getattr(model_type, 'create_model')
        self.model = create_model(self.config)
        print(self.model, end='\n\n')

        # Loss, Optimizer and LRScheduler
        self.criterion = getattr(loss_functions, self.config.loss_fn)(self.config)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config.learning_rate, alpha=0.95)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5,
                                                                    patience=3, verbose=True)
        self.early_stopping = EarlyStopping(patience=10)
        self.loss_agg = np.sum if config.loss_fn == 'WRMSSELevel12Loss' else np.mean

        # Metric
        self.metric = getattr(metrics, config.metric)()
        self.metric_2 = getattr(loss_functions, config.secondary_metric)()

        print(f' Loading Data '.center(self.terminal_width, '*'))
        data_loader = DataLoader(self.config)
        self.ids = data_loader.ids

        self.train_loader = data_loader.create_train_loader()
        self.val_loader = data_loader.create_val_loader()
        self.train_metric_t, self.train_metric_w, self.train_metric_s = data_loader.get_weights_and_scaling(
            self.config.training_ts['data_start_t'], self.config.training_ts['horizon_start_t'],
            self.config.training_ts['horizon_end_t'])
        self.val_metric_t, self.val_metric_w, self.val_metric_s = data_loader.get_weights_and_scaling(
            self.config.validation_ts['data_start_t'], self.config.validation_ts['horizon_start_t'],
            self.config.validation_ts['horizon_end_t'])
        self.n_windows = data_loader.n_windows

        self.start_epoch, self.min_val_error = 1, None
        # Load checkpoint if training is to be resumed
        self.model_checkpoint = ModelCheckpoint(config=self.config)
        if config.resume_training:
            self.model, self.optimizer, self.scheduler, [self.start_epoch, self.min_val_error, num_bad_epochs] = \
                self.model_checkpoint.load(self.model, self.optimizer, self.scheduler)
            self.early_stopping.best = self.min_val_error
            self.early_stopping.num_bad_epochs = num_bad_epochs
            print(f'Resuming model training from epoch {self.start_epoch}')
        else:
            # remove previous logs, if any
            if self.config.fold is None:
                logs = glob.glob('./logs/.*') + glob.glob('./logs/*')
                for f in logs:
                    try:
                        os.remove(f)
                    except IsADirectoryError:
                        shutil.rmtree(f)
            else:
                logs = glob.glob(f'./logs/fold_{self.config.fold}/.*') + glob.glob(f'./logs/fold_{self.config.fold}/*')
                for f in logs:
                    os.remove(f)

        # logging
        self.writer = SummaryWriter(f'logs') if self.config.fold is None \
            else SummaryWriter(f'logs/fold_{self.config.fold}')

    def _get_val_loss_and_err(self):
        self.model.eval()
        progbar = tqdm(self.val_loader)
        progbar.set_description("             ")
        losses, sec_metric, epoch_preds, epoch_ids, epoch_ids_idx = [], [], [], [], []
        for i, [x, y, norm_factor, ids_idx, loss_input, _] in enumerate(progbar):
            x = [inp.to(self.config.device) for inp in x]
            y = y.to(self.config.device)
            norm_factor = norm_factor.to(self.config.device)
            loss_input = [inp.to(self.config.device) for inp in loss_input]
            epoch_ids.append(self.ids[ids_idx])
            epoch_ids_idx.append(ids_idx.numpy())

            preds = self.model(*x) * norm_factor[:, None]
            epoch_preds.append(preds.data.cpu().numpy())
            loss = self.criterion(preds, y, *loss_input)
            losses.append(loss.data.cpu().numpy())
            sec_metric.append(self.metric_2(preds, y, *loss_input).data.cpu().numpy())

        # Sort to remove shuffle applied by the dataset loader
        sort_idx = np.argsort(np.concatenate(epoch_ids_idx, axis=0))
        epoch_preds = np.concatenate(epoch_preds, axis=0)[sort_idx]
        epoch_ids = np.concatenate(epoch_ids, axis=0)[sort_idx]
        validation_agg_preds, _, _ = get_aggregated_series(epoch_preds, epoch_ids)
        val_error = self.metric.get_error(validation_agg_preds, self.val_metric_t, self.val_metric_s, self.val_metric_w)

        return self.loss_agg(losses), val_error, np.mean(sec_metric)

    def train(self):
        print(f' Training '.center(self.terminal_width, '*'), end='\n\n')

        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            print(f' Epoch [{epoch}/{self.config.num_epochs}] '.center(self.terminal_width, 'x'))
            self.model.train()
            progbar = tqdm(self.train_loader)
            losses, sec_metric, epoch_preds, epoch_ids, epoch_ids_idx = [], [], [], [], []

            for i, [x, y, norm_factor, ids_idx, loss_input, window_id] in enumerate(progbar):
                x = [inp.to(self.config.device) for inp in x]
                y = y.to(self.config.device)
                norm_factor = norm_factor.to(self.config.device)
                loss_input = [inp.to(self.config.device) for inp in loss_input]

                # Forward + Backward + Optimize
                self.optimizer.zero_grad()
                preds = self.model(*x) * norm_factor[:, None]

                if self.config.sliding_window:
                    if torch.sum(window_id == self.n_windows - 1) > 0:
                        epoch_ids.append(self.ids[ids_idx[window_id == self.n_windows - 1]].reshape(-1, 5))
                        epoch_ids_idx.append(ids_idx[window_id == self.n_windows - 1].numpy())
                        epoch_preds.append(preds[window_id == self.n_windows - 1].data.cpu().numpy().reshape(-1, 28))
                else:
                    epoch_ids.append(self.ids[ids_idx])
                    epoch_ids_idx.append(ids_idx.numpy())
                    epoch_preds.append(preds.data.cpu().numpy())

                sec_metric.append(self.metric_2(preds, y, *loss_input).data.cpu().numpy())

                loss = self.criterion(preds, y, *loss_input)
                losses.append(loss.data.cpu().numpy())

                if self.config.loss_fn == 'WRMSSELevel12Loss':
                    progbar.set_description("loss = %0.3f " % np.round(
                        (len(self.train_loader) / (i + 1)) * self.loss_agg(losses) / self.n_windows, 3))
                else:
                    progbar.set_description("loss = %0.3f " % np.round(self.loss_agg(losses), 3))

                loss.backward()
                self.optimizer.step()

            # Get training and validation loss and error
            # Sort to remove shuffle applied by the dataset loader
            sort_idx = np.argsort(np.concatenate(epoch_ids_idx, axis=0))
            epoch_preds = np.concatenate(epoch_preds, axis=0)[sort_idx]
            epoch_ids = np.concatenate(epoch_ids, axis=0)[sort_idx]
            training_agg_preds, _, _ = get_aggregated_series(epoch_preds, epoch_ids)
            if self.config.loss_fn == 'WRMSSELevel12Loss':
                train_loss = self.loss_agg(losses) / self.n_windows
            else:
                train_loss = self.loss_agg(losses)
            train_error = self.metric.get_error(training_agg_preds, self.train_metric_t,
                                                self.train_metric_s, self.train_metric_w)
            train_error_2 = np.mean(sec_metric)

            val_loss, val_error, val_error_2 = self._get_val_loss_and_err()

            print(f'Training Loss: {train_loss:.4f}, Training Error: {train_error:.4f}, '
                  f'Training Secondary Error: {train_error_2:.4f}\n'
                  f'Validation Loss: {val_loss:.4f}, Validation Error: {val_error:.4f}, '
                  f'Validation Secondary Error: {val_error_2:.4f}')

            # Change learning rate according to scheduler
            self.scheduler.step(val_error)

            # save checkpoint and best model
            if self.min_val_error is None:
                self.min_val_error = val_error
                is_best = True
                print(f'Best model obtained at the end of epoch {epoch}')
            else:
                if val_error < self.min_val_error:
                    self.min_val_error = val_error
                    is_best = True
                    print(f'Best model obtained at the end of epoch {epoch}')
                else:
                    is_best = False
            self.model_checkpoint.save(is_best, self.min_val_error, self.early_stopping.num_bad_epochs,
                                       epoch, self.model, self.optimizer, self.scheduler)

            # write logs
            self.writer.add_scalar(f'{self.config.loss_fn}/train', train_loss, epoch * i)
            self.writer.add_scalar(f'{self.config.loss_fn}/val', val_loss, epoch * i)
            self.writer.add_scalar(f'{self.config.metric}/train', train_error, epoch * i)
            self.writer.add_scalar(f'{self.config.metric}/val', val_error, epoch * i)
            self.writer.add_scalar(f'{self.config.secondary_metric}/train', train_error_2, epoch * i)
            self.writer.add_scalar(f'{self.config.secondary_metric}/val', val_error_2, epoch * i)

            # Early Stopping
            if self.early_stopping.step(val_error):
                print(f' Training Stopped'.center(self.terminal_width, '*'))
                print(f'Early stopping triggered after epoch {epoch}')
                break

        self.writer.close()
        
