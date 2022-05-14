import torch


class Config:

    resume_training = False
    resume_from_fold = 1  # In case of k-fold training [1, k]

    loss_fn = 'WRMSSELevel12Loss'
    metric = 'WRMSSEMetric'
    secondary_metric = 'RMSSELoss'
    architecture = 'seq2seq_w_attn_on_hid'

    # Running a sliding window training will help increase the training data
    sliding_window = True  # Note: sliding window has not been tested with WRMSSELoss
    window_length = 28 * 13

    lag_and_roll_feats = True  # Note: Currently only works with dilated_seq2seq & seq2seq_w_attn_on_hid architectures
    lags = list(range(27, 42))
    rolling = [7, 14, 30, 60, 180]

    # Regularization
    add_random_noise = True
    noise_rate = 0.5

    # *** RNN *** #
    # hidden dimension and no. of layers will be the same for both encoder and decoder
    rnn_num_hidden = 128
    rnn_num_layers = 2
    bidirectional = True
    enc_rnn_dropout = 0.2
    dec_rnn_dropout = 0.0
    teacher_forcing_ratio = 0.0

    num_epochs = 200
    batch_size = 512
    learning_rate = 0.0003

    # training, validation and test periods
    training_ts = {'data_start_t': 1969 - 1 - (28 * 29), 'horizon_start_t': 1969 - 1 - (28 * 3),
                   'horizon_end_t': 1969 - 1 - (28 * 2)}
    validation_ts = {'data_start_t': 1969 - 1 - (28 * 15), 'horizon_start_t': 1969 - 1 - (28 * 2),
                     'horizon_end_t': 1969 - 1 - (28 * 1)}
    test_ts = {'data_start_t': 1969 - 1 - (28 * 14), 'horizon_start_t': 1969 - 1 - (28 * 1),
               'horizon_end_t': 1969 - 1 - (28 * 0)}

    # Parameters for k-fold training
    k_fold = True
    k_fold_splits = [(f_train_ts, f_val_ts) for f_train_ts, f_val_ts in
                     zip([
                         {'data_start_t': 1969 - 1 - (28 * 31), 'horizon_start_t': 1969 - 1 - (28 * 5),
                          'horizon_end_t': 1969 - 1 - (28 * 4)},
                         {'data_start_t': 1969 - 1 - (28 * 30), 'horizon_start_t': 1969 - 1 - (28 * 4),
                          'horizon_end_t': 1969 - 1 - (28 * 3)},
                         {'data_start_t': 1969 - 1 - (28 * 29), 'horizon_start_t': 1969 - 1 - (28 * 3),
                          'horizon_end_t': 1969 - 1 - (28 * 2)}
                     ], [
                         {'data_start_t': 1969 - 1 - (28 * 17), 'horizon_start_t': 1969 - 1 - (28 * 4),
                          'horizon_end_t': 1969 - 1 - (28 * 3)},
                         {'data_start_t': 1969 - 1 - (28 * 16), 'horizon_start_t': 1969 - 1 - (28 * 3),
                          'horizon_end_t': 1969 - 1 - (28 * 2)},
                         {'data_start_t': 1969 - 1 - (28 * 15), 'horizon_start_t': 1969 - 1 - (28 * 2),
                          'horizon_end_t': 1969 - 1 - (28 * 1)}
                     ])]

    data_file = './processed_data/data.pickle'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
