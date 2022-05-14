import sys

sys.path.extend(['..'])

import torch
import torch.utils.data
import torch.utils.data as data_utils
import pickle as pkl

from utils.data_utils import *


# Dataset (Input Pipeline)
class CustomDataset(data_utils.Dataset):
    """
    Custom dataset

    Let:
    training period timesteps = [0, N]
    prediction period timesteps = [N+1, N+P]

    Arguments:
    X_prev_day_sales : previous day sales for training period ([0, N])
    X_enc_only_feats : aggregated series' previous day sales for training period ([0, N])
    X_enc_dec_feats : sell price and categorical features for training and prediction period ([0, N+P])
    X_calendar : calendar features for training and prediction period ([0, N+P])
    X_last_day_sales : the actual sales for the day before the start of the prediction period (for timestep N)
                       (this will serve as the first timestep's input for the decoder)
    Y : actual sales, denoting targets for prediction period ([N+1, N+P])

    Returns:
    List of torch arrays:
    x_enc: concatenated encoder features (except embedding)
    x_enc_emb: concatenated encoder embedding features
    x_dec: concatenated decoder features (except embedding)
    x_dec_emb: concatenated decoder embedding features
    x_last_day_sales: the actual sales for the day before the start of the prediction period
    y: targets (only in training phase)
    """

    def __init__(self, X_prev_day_sales, X_enc_only_feats, X_enc_dec_feats, X_calendar, norm_factor, norm_factor_sell_p,
                 window_time_range, lagged_feats=None, rolling_feats=None, Y=None, rmsse_denominator=None,
                 wrmsse_weights=None, window_id=None, config=None, is_training=True):

        self.X_prev_day_sales = X_prev_day_sales
        self.X_enc_only_feats = X_enc_only_feats
        self.X_enc_dec_feats = X_enc_dec_feats
        self.X_calendar = X_calendar
        self.norm_factor = norm_factor
        self.norm_factor_sell_p = norm_factor_sell_p
        self.window_time_range = window_time_range
        self.window_id = window_id
        self.lagged_feats = lagged_feats
        self.rolling_feats = rolling_feats
        self.config = config
        self.is_training = is_training

        if Y is not None:
            self.Y = torch.from_numpy(Y).float()
            self.rmsse_denominator = torch.from_numpy(rmsse_denominator).float()
            self.wrmsse_weights = torch.from_numpy(wrmsse_weights).float()
        else:
            self.Y = None

    def __len__(self):
        return self.norm_factor.shape[0]

    def __getitem__(self, idx):
        if self.window_id is not None:
            time_range = self.window_time_range[self.window_id[idx]]
            scale = self.rmsse_denominator[idx - (self.window_id[idx] * 30490)]
            weight = self.wrmsse_weights[idx - (self.window_id[idx] * 30490)]
            ids_idx = idx - (self.window_id[idx] * 30490)
            window_id = self.window_id[idx]
        else:
            time_range = self.window_time_range
            ids_idx = idx
            window_id = 0
            if self.Y is not None:
                scale = self.rmsse_denominator[idx]
                weight = self.wrmsse_weights[idx]

        # Filter data for time range of the selected window, also normalize prev_day_sales and sell_price
        norm_factor = self.norm_factor[idx]
        X_calendar = self.X_calendar[time_range[0]:time_range[2]]

        X_prev_day_sales = self.X_prev_day_sales[time_range[0]:time_range[1], ids_idx] / norm_factor
        X_prev_day_sales_dec = self.X_prev_day_sales[time_range[1]:time_range[2], ids_idx] / norm_factor
        X_prev_day_sales[X_prev_day_sales < 0] = -1.0
        X_prev_day_sales_dec[X_prev_day_sales_dec < 0] = -1.0

        if self.lagged_feats is not None:
            X_lag_feats_enc = self.lagged_feats[time_range[0]:time_range[1], ids_idx] / norm_factor
            X_lag_feats_dec = self.lagged_feats[time_range[1]:time_range[2], ids_idx] / norm_factor
            X_lag_feats_enc[X_lag_feats_enc < 0] = -1.0
            X_lag_feats_dec[X_lag_feats_dec < 0] = -1.0
            # rolling features for decoder will be calculated on the fly (by including predictions for the prev steps)
            X_roll_feats_enc = self.rolling_feats[time_range[0]:time_range[1], ids_idx] / norm_factor

        # If training and if enabled in config, multiply sales features by random noise
        # (new value will be lower bound by 0)
        if self.config.add_random_noise and self.is_training:
            if len(X_prev_day_sales[X_prev_day_sales >= 0]) > 0:
                random_noise = np.clip(np.random.normal(1, X_prev_day_sales[X_prev_day_sales >= 0].std(),
                                                        time_range[2] - time_range[0]), 0, None)
                noise = np.ones_like(random_noise)
                mask = np.random.choice([0, 1], size=noise.shape, p=((1 - self.config.noise_rate),
                                                                     self.config.noise_rate)).astype(np.bool)
                noise[mask] = random_noise[mask]

                X_prev_day_sales[X_prev_day_sales >= 0] *= noise[:time_range[1] - time_range[0]][X_prev_day_sales >= 0]
                X_prev_day_sales_dec[X_prev_day_sales_dec >= 0] *= noise[time_range[1]
                                                                         - time_range[2]:][X_prev_day_sales_dec >= 0]

            if self.lagged_feats is not None:
                # lagged features
                if len(X_lag_feats_enc[X_lag_feats_enc >= 0]) > 0:
                    random_noise = np.clip(np.random.normal(1, X_lag_feats_enc[X_lag_feats_enc >= 0].std(0),
                                                            [time_range[2] - time_range[0],
                                                             X_lag_feats_enc.shape[1]]), 0, None)
                    noise = np.ones_like(random_noise)
                    mask = np.random.choice([0, 1], size=noise.shape, p=((1 - self.config.noise_rate),
                                                                         self.config.noise_rate)).astype(np.bool)
                    noise[mask] = random_noise[mask]

                    X_lag_feats_enc[X_lag_feats_enc >= 0] *= noise[:time_range[1] - time_range[0]][X_lag_feats_enc >= 0]
                    X_lag_feats_dec[X_lag_feats_dec >= 0] *= noise[time_range[1]
                                                                   - time_range[2]:][X_lag_feats_dec >= 0]
                
                # rolling features
                random_noise = np.clip(np.random.normal(1, X_roll_feats_enc[:, :len(self.config.rolling)].std(0),
                                                        [time_range[1] - time_range[0],
                                                         len(self.config.rolling)]), 0, None)
                noise = np.ones_like(random_noise)
                mask = np.random.choice([0, 1], size=noise.shape, p=((1 - self.config.noise_rate),
                                                                     self.config.noise_rate)).astype(np.bool)
                noise[mask] = random_noise[mask]

                X_roll_feats_enc[:, :len(self.config.rolling)] *= noise
                X_roll_feats_enc[:, len(self.config.rolling):] *= noise

        X_enc_dec_feats = self.X_enc_dec_feats[time_range[0]:time_range[2], ids_idx]

        # Directly dividing the sell price column leads to memory explosion
        norm_factor_sell_p = np.ones_like(X_enc_dec_feats, np.float64)
        norm_factor_sell_p[:, 0] = self.norm_factor_sell_p[idx]
        X_enc_dec_feats = X_enc_dec_feats / norm_factor_sell_p

        if self.Y is not None:
            Y = self.Y[ids_idx, time_range[1]:time_range[2]]

        enc_timesteps = time_range[1] - time_range[0]
        dec_timesteps = time_range[2] - time_range[0] - enc_timesteps
        num_embedding = 5
        num_cal_embedding = 2

        # input data for encoder
        x_enc_dec_feats_enc = X_enc_dec_feats[:enc_timesteps, :-num_embedding].reshape(enc_timesteps, -1)

        x_prev_day_sales_enc = X_prev_day_sales.reshape(-1, 1)
        x_sales_feats_enc = x_prev_day_sales_enc if self.lagged_feats is None \
            else np.concatenate([x_prev_day_sales_enc, X_lag_feats_enc, X_roll_feats_enc], 1)
        x_calendar_enc = X_calendar[:enc_timesteps, :-num_cal_embedding]
        x_calendar_enc_emb = X_calendar[:enc_timesteps, -num_cal_embedding:].reshape(enc_timesteps, -1)

        x_enc = np.concatenate([x_enc_dec_feats_enc, x_calendar_enc, x_sales_feats_enc], axis=1)
        x_enc_emb = X_enc_dec_feats[:enc_timesteps, -num_embedding:].reshape(enc_timesteps, -1)

        # input data for decoder
        x_enc_dec_feats_dec = X_enc_dec_feats[enc_timesteps:, :-num_embedding].reshape(dec_timesteps, -1)
        x_calendar_dec = X_calendar[enc_timesteps:, :-num_cal_embedding]
        x_calendar_dec_emb = X_calendar[enc_timesteps:, -num_cal_embedding:].reshape(dec_timesteps, -1)
        
        x_prev_day_sales_dec = X_prev_day_sales_dec.reshape(-1, 1)
        x_sales_feats_dec = x_prev_day_sales_dec if self.lagged_feats is None \
            else np.concatenate([x_prev_day_sales_dec, X_lag_feats_dec], 1)

        x_dec = np.concatenate([x_enc_dec_feats_dec, x_calendar_dec], axis=1)
        x_dec_emb = X_enc_dec_feats[enc_timesteps:, -num_embedding:].reshape(dec_timesteps, -1)

        if self.Y is None:
            return [[torch.from_numpy(x_enc).float(), torch.from_numpy(x_enc_emb).long(),
                     torch.from_numpy(x_calendar_enc_emb).long(),
                     torch.from_numpy(x_dec).float(), torch.from_numpy(x_dec_emb).long(),
                     torch.from_numpy(x_calendar_dec_emb).long(),
                     torch.from_numpy(x_sales_feats_dec).float()], norm_factor]

        return [[torch.from_numpy(x_enc).float(), torch.from_numpy(x_enc_emb).long(),
                 torch.from_numpy(x_calendar_enc_emb).long(),
                 torch.from_numpy(x_dec).float(), torch.from_numpy(x_dec_emb).long(),
                 torch.from_numpy(x_calendar_dec_emb).long(),
                 torch.from_numpy(x_sales_feats_dec).float()],
                Y, torch.from_numpy(np.array(norm_factor)).float(),
                ids_idx,
                [scale, weight],
                window_id]


class DataLoader:
    def __init__(self, config):
        self.config = config

        # load data
        with open(f'{self.config.data_file}', 'rb') as f:
            data_dict = pkl.load(f)

        self.ids = data_dict['sales_data_ids']
        self.enc_dec_feat_names = data_dict['enc_dec_feat_names']
        self.sell_price_i = self.enc_dec_feat_names.index('sell_price')
        self.X_prev_day_sales = data_dict['X_prev_day_sales']
        self.X_enc_only_feats = data_dict['X_enc_only_feats']
        self.X_enc_dec_feats = data_dict['X_enc_dec_feats']
        self.X_calendar = data_dict['X_calendar']
        self.enc_dec_feat_names = data_dict['enc_dec_feat_names']
        self.Y = data_dict['Y']

        # for prev_day_sales, set value as -1 for the period the product was not actively sold
        self.X_prev_day_sales_unsold_negative = self.X_prev_day_sales.copy()
        
        for idx, first_non_zero_idx in enumerate((self.X_prev_day_sales != 0).argmax(axis=0)):
            self.X_prev_day_sales_unsold_negative[:first_non_zero_idx, idx] = -1

        self.n_windows = 1

    def create_train_loader(self, data_start_t=None, horizon_start_t=None, horizon_end_t=None):
        if (data_start_t is None) | (horizon_start_t is None) | (horizon_end_t is None):
            data_start_t = self.config.training_ts['data_start_t']
            horizon_start_t = self.config.training_ts['horizon_start_t']
            horizon_end_t = self.config.training_ts['horizon_end_t']

        # Run a sliding window of length "window_length" and train for the next month of each window
        if self.config.sliding_window:
            window_length = self.config.window_length
            window_time_range, norm_factor, norm_factor_sell_p = [], [], []
            weights, scales = [], []

            for idx, i in enumerate(range(data_start_t + window_length, horizon_end_t, 28)):
                w_data_start_t, w_horizon_start_t = data_start_t + (idx * 28), i
                w_horizon_end_t = w_horizon_start_t + 28
                window_time_range.append([w_data_start_t - data_start_t, w_horizon_start_t - data_start_t,
                                          w_horizon_end_t - data_start_t])

                # calculate denominator for rmsse loss
                squared_movement = ((self.Y.T[:w_horizon_start_t] -
                                     self.X_prev_day_sales[:w_horizon_start_t]).astype(np.int64) ** 2)
                actively_sold_in_range = (self.X_prev_day_sales[:w_horizon_start_t] != 0).argmax(axis=0)
                rmsse_den = []
                for idx_active_sell, first_active_sell_idx in enumerate(actively_sold_in_range):
                    den = squared_movement[first_active_sell_idx:, idx_active_sell].mean()
                    den = den if den != 0 else 1
                    rmsse_den.append(den)
                scales.append(np.array(rmsse_den))

                # Get weights for WRMSSE and SPL loss
                w_weights = get_weights_level_12(self.Y[:, w_horizon_start_t - 28:w_horizon_start_t],
                                                 self.X_enc_dec_feats[w_horizon_start_t - 28:w_horizon_start_t, :,
                                                 self.sell_price_i].T)
                weights.append(w_weights)

                # Normalize sale features by dividing by mean of each series (as per the selected input window)
                w_X_prev_day_sales_calc = self.X_prev_day_sales[w_data_start_t:w_horizon_start_t]
                w_norm_factor = np.mean(w_X_prev_day_sales_calc, 0)
                w_norm_factor[w_norm_factor == 0] = 1.

                w_X_sell_p = self.X_enc_dec_feats[w_data_start_t:w_horizon_start_t, :, self.sell_price_i].copy().astype(
                    float)
                w_norm_factor_sell_p = np.median(w_X_sell_p, 0)
                w_norm_factor_sell_p[w_norm_factor_sell_p == 0] = 1.
                norm_factor.append(w_norm_factor)
                norm_factor_sell_p.append(w_norm_factor_sell_p)

            self.n_windows = idx + 1
            scales = np.concatenate(scales, 0)
            weights = np.concatenate(weights, 0)
            norm_factor = np.concatenate(norm_factor, 0)
            norm_factor_sell_p = np.concatenate(norm_factor_sell_p, 0)
            window_time_range = np.array(window_time_range)
            window_id = np.arange(idx + 1).repeat(self.X_enc_dec_feats.shape[1])

        else:
            # calculate denominator for rmsse loss
            squared_movement = ((self.Y.T[:horizon_start_t] -
                                 self.X_prev_day_sales[:horizon_start_t]).astype(np.int64) ** 2)
            actively_sold_in_range = (self.X_prev_day_sales[:horizon_start_t] != 0).argmax(axis=0)
            rmsse_den = []
            for idx_active_sell, first_active_sell_idx in enumerate(actively_sold_in_range):
                den = squared_movement[first_active_sell_idx:, idx_active_sell].mean()
                den = den if den != 0 else 1
                rmsse_den.append(den)

            # Get weights for WRMSSE and SPL loss
            weights = get_weights_level_12(self.Y[:, horizon_start_t - 28:horizon_start_t],
                                           self.X_enc_dec_feats[horizon_start_t - 28:horizon_start_t, :,
                                           self.sell_price_i].T)

            # Normalize sale features by dividing by mean of each series (as per the selected input window)
            X_prev_day_sales_calc = self.X_prev_day_sales[data_start_t:horizon_start_t]
            norm_factor = np.mean(X_prev_day_sales_calc, 0)
            norm_factor[norm_factor == 0] = 1.

            X_sell_p = self.X_enc_dec_feats[data_start_t:horizon_start_t, :, self.sell_price_i].copy().astype(float)
            norm_factor_sell_p = np.median(X_sell_p, 0)
            norm_factor_sell_p[norm_factor_sell_p == 0] = 1.

            window_time_range = np.array([0, horizon_start_t - data_start_t, horizon_end_t - data_start_t])
            scales = np.array(rmsse_den)
            window_id = None

        # Add rolling and lag features
        if self.config.lag_and_roll_feats:
            max_prev_ts_req = max(self.config.lags + self.config.rolling)
            lagged_feats = []
            for lag_i in np.array(sorted(self.config.lags, reverse=True)):
                lag_i_feat = np.roll(self.X_prev_day_sales_unsold_negative[data_start_t - max_prev_ts_req:]
                                     .astype(np.int32), lag_i, axis=0)
                lag_i_feat[:lag_i] = 0
                lagged_feats.append(lag_i_feat)
            lagged_feats = np.stack(lagged_feats, axis=2)[max_prev_ts_req:]

            rolling_feats, roll_i_means, roll_i_stds = [], [], []
            roll_df = pd.DataFrame(self.X_prev_day_sales[data_start_t - max_prev_ts_req:].astype(np.int32))
            for roll_i in self.config.rolling:
                roll_i_feat_mean = pd.DataFrame(roll_df).rolling(roll_i, axis=0).mean().fillna(0).values
                roll_i_means.append(roll_i_feat_mean)
            for roll_i in self.config.rolling:
                roll_i_feat_std = pd.DataFrame(roll_df).rolling(roll_i, axis=0).std().fillna(0).values
                roll_i_stds.append(roll_i_feat_std)
            rolling_feats = np.stack(roll_i_means + roll_i_stds, 2)[max_prev_ts_req:]
            
        else:
            lagged_feats, rolling_feats = None, None
        
        dataset = CustomDataset(self.X_prev_day_sales_unsold_negative[data_start_t:],
                                self.X_enc_only_feats[data_start_t:],
                                self.X_enc_dec_feats[data_start_t:],
                                self.X_calendar[data_start_t:],
                                norm_factor, norm_factor_sell_p, window_time_range,
                                lagged_feats, rolling_feats,
                                Y=self.Y[:, data_start_t:],
                                rmsse_denominator=scales, wrmsse_weights=weights, window_id=window_id,
                                config=self.config)

        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.batch_size, shuffle=False,
                                           num_workers=6, pin_memory=True)

    def create_val_loader(self, data_start_t=None, horizon_start_t=None, horizon_end_t=None):
        if (data_start_t is None) | (horizon_start_t is None) | (horizon_end_t is None):
            data_start_t = self.config.validation_ts['data_start_t']
            horizon_start_t = self.config.validation_ts['horizon_start_t']
            horizon_end_t = self.config.validation_ts['horizon_end_t']

        # calculate denominator for rmsse loss
        squared_movement = ((self.Y.T[:horizon_start_t] -
                             self.X_prev_day_sales[:horizon_start_t]).astype(np.int64) ** 2)
        actively_sold_in_range = (self.X_prev_day_sales[:horizon_start_t] != 0).argmax(axis=0)
        rmsse_den = []
        for idx, first_active_sell_idx in enumerate(actively_sold_in_range):
            den = squared_movement[first_active_sell_idx:, idx].mean()
            den = den if den != 0 else 1
            rmsse_den.append(den)

        # Get weights for WRMSSE and SPL loss
        weights = get_weights_level_12(self.Y[:, horizon_start_t-28:horizon_start_t],
                                       self.X_enc_dec_feats[horizon_start_t-28:horizon_start_t, :, self.sell_price_i].T)

        # Normalize sale features by dividing by mean of each series (as per the selected input window)
        X_prev_day_sales_calc = self.X_prev_day_sales[data_start_t:horizon_start_t]
        norm_factor = np.mean(X_prev_day_sales_calc, 0)
        norm_factor[norm_factor == 0] = 1.

        X_sell_p = self.X_enc_dec_feats[data_start_t:horizon_start_t, :, self.sell_price_i].copy().astype(float)
        norm_factor_sell_p = np.median(X_sell_p, 0)
        norm_factor_sell_p[norm_factor_sell_p == 0] = 1.

        window_time_range = [0, horizon_start_t - data_start_t, horizon_end_t - data_start_t]

        # Add rolling and lag features
        if self.config.lag_and_roll_feats:
            max_prev_ts_req = max(self.config.lags + self.config.rolling)
            lagged_feats = []
            for lag_i in np.array(sorted(self.config.lags, reverse=True)):
                lag_i_feat = np.roll(self.X_prev_day_sales_unsold_negative[data_start_t - max_prev_ts_req:]
                                     .astype(np.int32), lag_i, axis=0)
                lag_i_feat[:lag_i] = 0
                lagged_feats.append(lag_i_feat)
            lagged_feats = np.stack(lagged_feats, axis=2)[max_prev_ts_req:]

            rolling_feats, roll_i_means, roll_i_stds = [], [], []
            roll_df = pd.DataFrame(self.X_prev_day_sales[data_start_t - max_prev_ts_req:].astype(np.int32))
            for roll_i in self.config.rolling:
                roll_i_feat_mean = pd.DataFrame(roll_df).rolling(roll_i, axis=0).mean().fillna(0).values
                roll_i_means.append(roll_i_feat_mean)
            for roll_i in self.config.rolling:
                roll_i_feat_std = pd.DataFrame(roll_df).rolling(roll_i, axis=0).std().fillna(0).values
                roll_i_stds.append(roll_i_feat_std)
            rolling_feats = np.stack(roll_i_means + roll_i_stds, 2)[max_prev_ts_req:]
        else:
            lagged_feats, rolling_feats = None, None

        dataset = CustomDataset(self.X_prev_day_sales_unsold_negative[data_start_t:],
                                self.X_enc_only_feats[data_start_t:],
                                self.X_enc_dec_feats[data_start_t:],
                                self.X_calendar[data_start_t:],
                                norm_factor, norm_factor_sell_p, window_time_range,
                                lagged_feats, rolling_feats,
                                Y=self.Y[:, data_start_t:],
                                rmsse_denominator=np.array(rmsse_den), wrmsse_weights=weights,
                                config=self.config, is_training=False)

        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.batch_size, num_workers=6,
                                           pin_memory=True)

    def create_test_loader(self, data_start_t=None, horizon_start_t=None, horizon_end_t=None):
        if (data_start_t is None) | (horizon_start_t is None) | (horizon_end_t is None):
            data_start_t = self.config.test_ts['data_start_t']
            horizon_start_t = self.config.test_ts['horizon_start_t']
            horizon_end_t = self.config.test_ts['horizon_end_t']

        # Normalize sale features by dividing by mean of each series (as per the selected input window)
        X_prev_day_sales_calc = self.X_prev_day_sales[data_start_t:horizon_start_t]
        norm_factor = np.mean(X_prev_day_sales_calc, 0)
        norm_factor[norm_factor == 0] = 1.

        X_sell_p = self.X_enc_dec_feats[data_start_t:horizon_start_t, :, self.sell_price_i].copy().astype(float)
        norm_factor_sell_p = np.median(X_sell_p, 0)
        norm_factor_sell_p[norm_factor_sell_p == 0] = 1.

        window_time_range = [0, horizon_start_t - data_start_t, horizon_end_t - data_start_t]

        # Add rolling and lag features
        if self.config.lag_and_roll_feats:
            max_prev_ts_req = max(self.config.lags + self.config.rolling)
            lagged_feats = []
            for lag_i in np.array(sorted(self.config.lags, reverse=True)):
                lag_i_feat = np.roll(self.X_prev_day_sales_unsold_negative[data_start_t - max_prev_ts_req:]
                                     .astype(np.int32), lag_i, axis=0)
                lag_i_feat[:lag_i] = 0
                lagged_feats.append(lag_i_feat)
            lagged_feats = np.stack(lagged_feats, axis=2)[max_prev_ts_req:]

            rolling_feats, roll_i_means, roll_i_stds = [], [], []
            roll_df = pd.DataFrame(self.X_prev_day_sales[data_start_t - max_prev_ts_req:].astype(np.int32))
            for roll_i in self.config.rolling:
                roll_i_feat_mean = pd.DataFrame(roll_df).rolling(roll_i, axis=0).mean().fillna(0).values
                roll_i_means.append(roll_i_feat_mean)
            for roll_i in self.config.rolling:
                roll_i_feat_std = pd.DataFrame(roll_df).rolling(roll_i, axis=0).std().fillna(0).values
                roll_i_stds.append(roll_i_feat_std)
            
            rolling_feats = np.stack(roll_i_means + roll_i_stds, 2)[max_prev_ts_req:]
        else:
            lagged_feats, rolling_feats = None, None

        dataset = CustomDataset(self.X_prev_day_sales_unsold_negative[data_start_t:],
                                self.X_enc_only_feats[data_start_t:],
                                self.X_enc_dec_feats[data_start_t:],
                                self.X_calendar[data_start_t:],
                                norm_factor, norm_factor_sell_p, window_time_range,
                                lagged_feats, rolling_feats, config=self.config, is_training=False)

        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.batch_size, num_workers=6,
                                           pin_memory=True)

    def get_weights_and_scaling(self, data_start_t, horizon_start_t, horizon_end_t):
        """Returns aggregated target, weights and rmsse scaling factors for series of all 12 levels"""

        # Get aggregated series
        agg_series_Y, agg_series_id, _ = get_aggregated_series(self.Y[:, :horizon_end_t], self.ids)
        agg_target = agg_series_Y[:, horizon_start_t:]
        agg_series_Y = agg_series_Y[:, :horizon_start_t]
        agg_series_prev_day_sales, _, _ = get_aggregated_series(self.X_prev_day_sales.T[:, :horizon_start_t], self.ids)

        # calculate denominator for rmsse loss
        squared_movement = ((agg_series_Y.T - agg_series_prev_day_sales.T).astype(np.int64) ** 2)
        actively_sold_in_range = (agg_series_prev_day_sales.T != 0).argmax(axis=0)
        rmsse_den = []
        for idx, first_active_sell_idx in enumerate(actively_sold_in_range):
            den = squared_movement[first_active_sell_idx:, idx].mean()
            den = den if den != 0 else 1
            rmsse_den.append(den)

        # Get weights
        weights, _ = get_weights_all_levels(self.Y[:, horizon_start_t-28:horizon_start_t],
                                            self.X_enc_dec_feats[horizon_start_t-28:horizon_start_t, :,
                                            self.sell_price_i].T,
                                            self.ids)

        return agg_target, weights, np.array(rmsse_den)
