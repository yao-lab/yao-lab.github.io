import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from importlib import import_module
import os
import shutil
from glob import glob

from data_generator import DataLoader
from utils.training_utils import ModelCheckpoint
from config import Config


class SubmissionGenerator:
    def __init__(self, config):
        self.config = config
        self.terminal_width = shutil.get_terminal_size((80, 20)).columns

        # Model
        print(f' Model: {self.config.architecture} '.center(self.terminal_width, '*'), end='\n\n')
        model_type = import_module('models.' + self.config.architecture)
        create_model = getattr(model_type, 'create_model')

        if self.config.k_fold:
            self.model = []
            for fold in range(len(self.config.k_fold_splits)):
                self.config.fold = fold + 1
                model = create_model(self.config)
                model_checkpoint = ModelCheckpoint(config=self.config)
                model, _, _, _ = model_checkpoint.load(model, load_best=True)
                self.model.append(model)
            print(self.model[0])
        else:
            self.model = create_model(self.config)
            print(self.model)
            self.config.fold = None
            model_checkpoint = ModelCheckpoint(config=self.config)
            self.model, _, _, _ = model_checkpoint.load(self.model, load_best=True)

        print(f' Loading data '.center(self.terminal_width, '*'))
        data_loader = DataLoader(self.config)
        self.ids = data_loader.ids
        self.test_loader = data_loader.create_test_loader()


    def _generate_submission_file_model(self, model, outfile_prefix=''):
        print(f' Predict '.center(self.terminal_width, '*'))
        model.eval()
        preds = []
        for i, [x, norm_factor] in enumerate(tqdm(self.test_loader)):
            x = [inp.to(self.config.device) for inp in x]
            norm_factor = norm_factor.to(self.config.device)
            preds.append((model(*x) * norm_factor[:, None]).data.cpu().numpy())

        predictions = np.concatenate(preds, 0)
        sample_submission = pd.read_csv('./data/sample_submission_accuracy.csv')

        # Merge with sample_submission by using series ids
        pred_ids = np.array([f'{i[0]}_{i[3]}' for i in self.ids])
        predictions_df = pd.DataFrame(np.hstack([pred_ids.reshape(-1, 1), predictions]),
                                      columns=['id'] + [f'F{i + 1}' for i in range(28)])
        sample_submission = sample_submission[['id']].merge(predictions_df, how='left',
                                                            left_on=sample_submission.id.str[:-11], right_on='id')
        sample_submission['id'] = sample_submission['id_x']
        del sample_submission['id_x'], sample_submission['id_y']

        #sample_submission.to_csv(f'{outfile_prefix}submission.csv.gz', compression='gzip', index=False)
        return sample_submission.iloc[:,1:].to_numpy()

    def generate_submission_file(self):
        # If k-fold, use three fold models to create three separate submission files
        kfold_result = []
        index = pd.read_csv('./data/sample_submission_accuracy.csv')['id']
        columns = pd.read_csv('./data/sample_submission_accuracy.csv').iloc[:,1:].columns
        if self.config.k_fold:
            for fold in range(len(self.config.k_fold_splits)):
                print()
                print(f' Fold [{fold + 1}/{len(self.config.k_fold_splits)}] '.center(self.terminal_width, '*'))
                kfold_result.append(self._generate_submission_file_model(self.model[fold], f'fold_{fold + 1}_'))

            ensemble_result = np.array(kfold_result).astype(np.float32).mean(axis=0)
            ensemble_result = pd.DataFrame(ensemble_result, columns=columns)
            pd.concat([index, ensemble_result], axis=1).to_csv('submission_accuracy.csv', index=False)
        else:
            self._generate_submission_file_model(self.model)

