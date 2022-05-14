import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import random
from models.model_utils.drnn import DRNN
random.seed(0)


# Build a seq2seq model
# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_sizes, cal_embedding_sizes, config):
        super(Encoder, self).__init__()
        self.config = config
        self.input_size = input_size

        self.embeddings = nn.ModuleList([nn.Embedding(classes, hidden_size)
                                         for classes, hidden_size in embedding_sizes])
        self.cal_embedding = nn.Embedding(cal_embedding_sizes[0], cal_embedding_sizes[1])
        self.drnns = nn.ModuleList([DRNN(self.input_size, config.rnn_num_hidden, config.rnn_num_layers,
                                         dropout=config.enc_rnn_dropout, cell_type='LSTM')
                                    for i in range(config.bidirectional + 1)])

    def forward(self, x, x_emb, x_cal_emb):
        batch_size = x.shape[0]
        x, x_emb, x_cal_emb = x.permute(1, 0, 2), x_emb.permute(1, 0, 2), x_cal_emb.permute(1, 0, 2)  # make time-major
        output_emb = [emb(x_emb[:, :, i]) for i, emb in enumerate(self.embeddings)]
        output_emb = torch.cat(output_emb, 2)

        # share embedding layer for both the calendar events
        output_emb_cal = [self.cal_embedding(x_cal_emb[:, :, 0]), self.cal_embedding(x_cal_emb[:, :, 1])]
        output_emb_cal = torch.cat(output_emb_cal, 2)

        x_rnn = torch.cat([x, output_emb, output_emb_cal], 2)

        rnn_input = [x_rnn, torch.flip(x_rnn, [0])] if self.config.bidirectional else [x_rnn]
        drnn_outputs, drnn_h0, drnn_h1 = [], [], []
        for i, drnn in enumerate(self.drnns):
            last_output, [h0, h1] = drnn(rnn_input[i])
            drnn_outputs.append(last_output)
            drnn_h0.append([h.view(-1, batch_size, self.config.rnn_num_hidden) for h in h0])
            drnn_h1.append([h.view(-1, batch_size, self.config.rnn_num_hidden) for h in h1])
        drnn_hiddens = [drnn_h0, drnn_h1]

        if self.config.bidirectional:
            for j, drnn_h in enumerate(drnn_hiddens):
                drnn_hiddens[j] = [torch.cat([drnn_h[0][i], drnn_h[1][i]], 0)
                                   for i in range(self.config.rnn_num_layers)]
        else:
            drnn_hiddens = [drnn_hiddens[0][0], drnn_hiddens[1][0]]

        return drnn_outputs, drnn_hiddens


# Decoder
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_sizes, cal_embedding_sizes, output_size, config):
        super(Decoder, self).__init__()
        self.input_size = input_size

        self.embeddings = nn.ModuleList([nn.Embedding(classes, hidden_size)
                                         for classes, hidden_size in embedding_sizes])
        self.cal_embedding = nn.Embedding(cal_embedding_sizes[0], cal_embedding_sizes[1])
        self.rnn = nn.LSTM(self.input_size, config.rnn_num_hidden, config.rnn_num_layers,
                           bidirectional=config.bidirectional, dropout=config.dec_rnn_dropout)
        self.pred = nn.Linear(config.rnn_num_hidden * (config.bidirectional + 1), output_size)

    def forward(self, x, x_emb, x_cal_emb, hidden):
        x, x_emb, x_cal_emb = x.permute(1, 0, 2), x_emb.permute(1, 0, 2), x_cal_emb.permute(1, 0, 2)  # make time-major
        output_emb = [emb(x_emb[:, :, i]) for i, emb in enumerate(self.embeddings)]
        output_emb = torch.cat(output_emb, 2)

        # share embedding layer for both the calendar events
        output_emb_cal = [self.cal_embedding(x_cal_emb[:, :, 0]), self.cal_embedding(x_cal_emb[:, :, 1])]
        output_emb_cal = torch.cat(output_emb_cal, 2)

        x_rnn = torch.cat([x, output_emb, output_emb_cal], 2)

        output, hidden = self.rnn(x_rnn, hidden)
        output = F.relu(self.pred(output[0]))

        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, config):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        self.fc_h0 = nn.Linear(sum([2**i for i in range(config.rnn_num_layers)] * (config.bidirectional + 1)),
                               config.rnn_num_layers * (config.bidirectional + 1))
        self.fc_h1 = nn.Linear(sum([2**i for i in range(config.rnn_num_layers)] * (config.bidirectional + 1)),
                               config.rnn_num_layers * (config.bidirectional + 1))

    def forward(self, x_enc, x_enc_emb, x_cal_enc_emb, x_dec, x_dec_emb, x_cal_dec_emb, x_sales_feats_dec):
        batch_size, pred_len = x_dec.shape[0:2]

        # create a tensor to store the outputs
        predictions = torch.zeros(batch_size, pred_len).to(self.config.device)

        # Tensor for calculating rolling features for each decoder timestep
        prev_sales = x_enc[:, :, 11]

        encoder_output, hidden = self.encoder(x_enc, x_enc_emb, x_cal_enc_emb)
        h0 = self.fc_h0(torch.cat(hidden[0], 0).permute(1, 2, 0)).permute(2, 0, 1).contiguous()
        h1 = self.fc_h1(torch.cat(hidden[1], 0).permute(1, 2, 0)).permute(2, 0, 1).contiguous()
        hidden = [h0, h1]

        # for each prediction timestep, use the output of the previous step,
        # concatenated with other features as the input

        # enable teacher forcing only if model is in training phase
        use_teacher_forcing = True if (random.random() < self.config.teacher_forcing_ratio) & self.training else False

        for timestep in range(0, pred_len):

            if timestep == 0:
                # for the first timestep of decoder, use previous steps' sales
                dec_input = torch.cat([x_dec[:, 0, :], x_sales_feats_dec[:, 0]], dim=1).unsqueeze(1)
                prev_sales = torch.cat([prev_sales, x_sales_feats_dec[:, 0, 0].unsqueeze(1)], 1)
            else:
                if use_teacher_forcing:
                    dec_input = torch.cat([x_dec[:, timestep, :], x_sales_feats_dec[:, timestep],
                                           x_sales_feats_dec[:, timestep, 1:]], dim=1).unsqueeze(1)
                    prev_sales = torch.cat([prev_sales, x_sales_feats_dec[:, timestep, 0].unsqueeze(1)], 1)
                else:
                    # for next timestep, current timestep's output will serve as the input along with other features
                    dec_input = torch.cat([x_dec[:, timestep, :], decoder_output,
                                           x_sales_feats_dec[:, timestep, 1:]], dim=1).unsqueeze(1)
                    prev_sales = torch.cat([prev_sales, decoder_output], 1)

            # Create lagged and rolling features, if required
            if self.config.lag_and_roll_feats:
                # lagged features with a lag of <= 28 will be recreated to utilize predicted values
                update_lags = sorted(self.config.lags, reverse=True)
                update_lags = [l_i for l_i in update_lags if l_i <= 28]

                lagged_feats = []
                for lag_idx, lag_i in enumerate(update_lags):
                    lag_i_feat = prev_sales[:, -lag_i]
                    lagged_feats.append(lag_i_feat)
                lagged_feats = torch.stack(lagged_feats, 1).unsqueeze(1)

                rolling_feats_mean, rolling_feats_std = [], []
                for roll_idx, roll_i in enumerate(self.config.rolling):
                    roll_i_feat = prev_sales[:, -roll_i:]
                    rolling_feats_mean.append(roll_i_feat.mean(1))
                    rolling_feats_std.append(roll_i_feat.std(1))
                rolling_feats = torch.cat([torch.stack(rolling_feats_mean, 1), torch.stack(rolling_feats_std, 1)], 1)\
                    .unsqueeze(1)

                dec_input = torch.cat([dec_input, rolling_feats], 2) if len(update_lags) == 0 \
                    else torch.cat([dec_input[:, :, :-len(update_lags)], lagged_feats, rolling_feats], 2)

            # the hidden state of the encoder will be the initialize the decoder's hidden state
            decoder_output, hidden = self.decoder(dec_input, x_dec_emb[:, timestep, :].unsqueeze(1),
                                                  x_cal_dec_emb[:, timestep, :].unsqueeze(1), hidden)

            # add predictions to predictions tensor
            predictions[:, timestep] = decoder_output.view(-1)

        return predictions


def create_model(config):
    # for item_id, dept_id, cat_id, store_id, state_id respectively
    embedding_sizes = [(3049, 50), (7, 4), (3, 2), (10, 5), (3, 2)]
    cal_embedding_sizes = (31, 16)

    num_lag_roll_feats = len(config.lags) + (len(config.rolling) * 2) if config.lag_and_roll_feats else 0
    num_features_enc = 12 + sum([j for i, j in embedding_sizes]) + cal_embedding_sizes[1] * 2 + num_lag_roll_feats
    num_features_dec = 12 + sum([j for i, j in embedding_sizes]) + cal_embedding_sizes[1] * 2 + num_lag_roll_feats

    enc = Encoder(num_features_enc, embedding_sizes, cal_embedding_sizes, config)
    dec = Decoder(num_features_dec, embedding_sizes, cal_embedding_sizes, 1, config)
    model = Seq2Seq(enc, dec, config)
    model.to(config.device)

    return model
