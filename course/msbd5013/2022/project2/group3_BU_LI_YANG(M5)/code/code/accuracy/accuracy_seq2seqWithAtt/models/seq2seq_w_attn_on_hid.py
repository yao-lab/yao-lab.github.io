import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import random


# Build a seq2seq model
# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_sizes, cal_embedding_sizes, config):
        super(Encoder, self).__init__()
        self.input_size = input_size

        self.embeddings = nn.ModuleList([nn.Embedding(classes, hidden_size)
                                         for classes, hidden_size in embedding_sizes])
        self.cal_embedding = nn.Embedding(cal_embedding_sizes[0], cal_embedding_sizes[1])

        rnn_in_size = [self.input_size] + [config.rnn_num_hidden *
                                           (config.bidirectional + 1)] * (config.rnn_num_layers - 1)
        self.rnns = nn.ModuleList([nn.LSTM(rnn_in_size[i], config.rnn_num_hidden, 1, bidirectional=config.bidirectional)
                                   for i in range(config.rnn_num_layers)])
        self.rnn_dropouts = nn.ModuleList([nn.Dropout(config.enc_rnn_dropout) for i in range(config.rnn_num_layers-1)])

    def forward(self, x, x_emb, x_cal_emb):
        x, x_emb, x_cal_emb = x.permute(1, 0, 2), x_emb.permute(1, 0, 2), x_cal_emb.permute(1, 0, 2)  # make time-major
        output_emb = [emb(x_emb[:, :, i]) for i, emb in enumerate(self.embeddings)]
        output_emb = torch.cat(output_emb, 2)

        # share embedding layer for both the calendar events
        output_emb_cal = [self.cal_embedding(x_cal_emb[:, :, 0]), self.cal_embedding(x_cal_emb[:, :, 1])]
        output_emb_cal = torch.cat(output_emb_cal, 2)

        output = torch.cat([x, output_emb, output_emb_cal], 2)

        rnn_outputs, h_0, h_1 = [], [], []
        for i, rnn in enumerate(self.rnns):
            if i != 0:
                output = self.rnn_dropouts[i - 1](output)
            output, h = rnn(output)
            rnn_outputs.append(output)
            h_0.append(h[0].permute(1, 2, 0))
            h_1.append(h[1].permute(1, 2, 0))

        hidden = [torch.stack(h_0, 0), torch.stack(h_1, 0)]
        return rnn_outputs, hidden


# Attention Decoder
class AttnDecoder(nn.Module):
    def __init__(self, input_size, embedding_sizes, cal_embedding_sizes, output_size, config):
        super(AttnDecoder, self).__init__()
        self.config = config
        self.input_size = input_size
        self.max_length = config.window_length if config.sliding_window \
            else config.training_ts['horizon_start_t'] - config.training_ts['data_start_t']

        self.embeddings = nn.ModuleList([nn.Embedding(classes, hidden_size)
                                         for classes, hidden_size in embedding_sizes])
        self.cal_embedding = nn.Embedding(cal_embedding_sizes[0], cal_embedding_sizes[1])
        self.attns = nn.ModuleList([
            nn.Linear(self.input_size + (config.rnn_num_hidden * (config.bidirectional + 1) * 2), self.max_length)
            for i in range(config.rnn_num_layers)])
        self.attn_combine = nn.ModuleList([nn.Linear(2 * (config.rnn_num_hidden), config.rnn_num_hidden)
                                           for i in range(config.rnn_num_layers)])

        rnn_in_size = [self.input_size] + [config.rnn_num_hidden *
                                           (config.bidirectional + 1)] * (config.rnn_num_layers - 1)
        self.rnns = nn.ModuleList([nn.LSTM(rnn_in_size[i], config.rnn_num_hidden, 1, bidirectional=config.bidirectional)
                                   for i in range(config.rnn_num_layers)])
        self.rnn_dropouts = nn.ModuleList([nn.Dropout(config.dec_rnn_dropout) for i in range(config.rnn_num_layers-1)])
        self.pred = nn.Linear(config.rnn_num_hidden * (config.bidirectional + 1), output_size)

    def forward(self, x, x_emb, x_cal_emb, hidden, encoder_outputs):
        x, x_emb, x_cal_emb = x.permute(1, 0, 2), x_emb.permute(1, 0, 2), x_cal_emb.permute(1, 0, 2)  # make time-major
        output_emb = [emb(x_emb[:, :, i]) for i, emb in enumerate(self.embeddings)]
        output_emb = torch.cat(output_emb, 2)

        # share embedding layer for both the calendar events
        output_emb_cal = [self.cal_embedding(x_cal_emb[:, :, 0]), self.cal_embedding(x_cal_emb[:, :, 1])]
        output_emb_cal = torch.cat(output_emb_cal, 2)

        output = torch.cat([x, output_emb, output_emb_cal], 2)

        num_hidden = self.config.rnn_num_hidden * (self.config.bidirectional + 1)
        attn_weights = [F.softmax(attn(torch.cat((output[0], hidden[0][i].view(-1, num_hidden),
                                                  hidden[1][i].view(-1, num_hidden)), 1)), dim=1)
                        for i, attn in enumerate(self.attns)]
        attns_applied = [torch.bmm(attn_w.unsqueeze(1), encoder_outputs[i].permute(1, 0, 2))
                         for i, attn_w in enumerate(attn_weights)]

        dec_hidden = []
        for i, combine in enumerate(self.attn_combine):
            dec_h_0 = torch.cat((hidden[0][i].permute(0, 2, 1),
                                  attns_applied[i][:, 0, :].view(-1, self.config.bidirectional + 1, self.config.rnn_num_hidden)), 2)
            dec_h_1 = torch.cat((hidden[1][i].permute(0, 2, 1),
                                  attns_applied[i][:, 0, :].view(-1, self.config.bidirectional + 1, self.config.rnn_num_hidden)), 2)
            dec_h_0 = combine(dec_h_0).permute(1, 0, 2).contiguous()
            dec_h_1 = combine(dec_h_1).permute(1, 0, 2).contiguous()

            dec_hidden.append([dec_h_0, dec_h_1])

        h_0, h_1 = [], []
        for i, rnn in enumerate(self.rnns):
            if i != 0:
                output = self.rnn_dropouts[i - 1](output)
            output, h = rnn(output, dec_hidden[i])
            h_0.append(h[0].permute(1, 2, 0))
            h_1.append(h[1].permute(1, 2, 0))

        hidden = [torch.stack(h_0, 0), torch.stack(h_1, 0)]
        output = F.relu(self.pred(output[0]))

        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, config):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.max_length = config.window_length if config.sliding_window \
            else config.training_ts['horizon_start_t'] - config.training_ts['data_start_t']

    def forward(self, x_enc, x_enc_emb, x_cal_enc_emb, x_dec, x_dec_emb, x_cal_dec_emb, x_sales_feats_dec):
        batch_size, pred_len = x_dec.shape[0:2]

        # Ignore some initial timesteps of encoder data, according to the max_length allowed
        x_enc, x_enc_emb = x_enc[:, -self.max_length:], x_enc_emb[:, -self.max_length:]
        x_cal_enc_emb = x_cal_enc_emb[:, -self.max_length:]

        # create a tensor to store the outputs
        predictions = torch.zeros(batch_size, pred_len).to(self.config.device)

        # Tensor for calculating rolling features for each decoder timestep
        prev_sales = x_enc[:, :, 11]

        encoder_output, hidden = self.encoder(x_enc, x_enc_emb, x_cal_enc_emb)

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
            decoder_output, hidden, _ = self.decoder(dec_input, x_dec_emb[:, timestep, :].unsqueeze(1),
                                                     x_cal_dec_emb[:, timestep, :].unsqueeze(1), hidden,
                                                     encoder_output)

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
    dec = AttnDecoder(num_features_dec, embedding_sizes, cal_embedding_sizes, 1, config)
    model = Seq2Seq(enc, dec, config)
    model.to(config.device)

    return model
