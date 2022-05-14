import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import random
random.seed(0)


# Build a seq2seq model
# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_sizes, cal_embedding_sizes, config):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.seq_length = config.window_length if config.sliding_window \
            else config.training_ts['horizon_start_t'] - config.training_ts['data_start_t']

        self.embeddings = nn.ModuleList([nn.Embedding(classes, hidden_size)
                                         for classes, hidden_size in embedding_sizes])
        self.cal_embedding = nn.Embedding(cal_embedding_sizes[0], cal_embedding_sizes[1])
        self.fc = nn.Linear(self.input_size, self.input_size * 2)
        self.fc_1 = nn.Linear(self.input_size * 2, self.input_size)
        self.rnn = nn.LSTM(self.input_size * 3, config.rnn_num_hidden,
                           config.rnn_num_layers, dropout=config.enc_rnn_dropout, bidirectional=config.bidirectional)

    def forward(self, x, x_emb, x_cal_emb):
        x, x_emb, x_cal_emb = x.permute(1, 0, 2), x_emb.permute(1, 0, 2), x_cal_emb.permute(1, 0, 2)  # make time-major
        output_emb = [emb(x_emb[:, :, i]) for i, emb in enumerate(self.embeddings)]
        output_emb = torch.cat(output_emb, 2)

        # share embedding layer for both the calendar events
        output_emb_cal = [self.cal_embedding(x_cal_emb[:, :, 0]), self.cal_embedding(x_cal_emb[:, :, 1])]
        output_emb_cal = torch.cat(output_emb_cal, 2)

        x_rnn = torch.cat([x, output_emb, output_emb_cal], 2).permute(1, 0, 2)

        x_rnn = self.fc(x_rnn)
        x_rnn_1 = self.fc_1(x_rnn)
        x_rnn = torch.cat([x_rnn, x_rnn_1], 2).permute(1, 0, 2)
        output, hidden = self.rnn(x_rnn)

        return output, hidden


# Attention Decoder
class AttnDecoder(nn.Module):
    def __init__(self, input_size, embedding_sizes, cal_embedding_sizes, output_size, config):
        super(AttnDecoder, self).__init__()
        self.input_size = input_size
        self.max_length = config.window_length if config.sliding_window \
            else config.training_ts['horizon_start_t'] - config.training_ts['data_start_t']

        self.embeddings = nn.ModuleList([nn.Embedding(classes, hidden_size)
                                         for classes, hidden_size in embedding_sizes])
        self.cal_embedding = nn.Embedding(cal_embedding_sizes[0], cal_embedding_sizes[1])
        self.attn = nn.Linear(self.input_size + (config.rnn_num_hidden * 2), self.max_length)
        self.attn_combine = nn.Linear(self.input_size + (config.rnn_num_hidden * (config.bidirectional + 1)),
                                      self.input_size)
        self.fc = nn.Linear(self.input_size, self.input_size * 2)
        self.fc_1 = nn.Linear(self.input_size * 2, self.input_size)
        self.rnn = nn.LSTM(self.input_size * 3, config.rnn_num_hidden,
                           config.rnn_num_layers, dropout=config.dec_rnn_dropout, bidirectional=config.bidirectional)
        self.pred = nn.Linear(config.rnn_num_hidden * (config.bidirectional + 1), output_size)

    def forward(self, x, x_emb, x_cal_emb, hidden, encoder_outputs):
        x, x_emb, x_cal_emb = x.permute(1, 0, 2), x_emb.permute(1, 0, 2), x_cal_emb.permute(1, 0, 2)  # make time-major
        output_emb = [emb(x_emb[:, :, i]) for i, emb in enumerate(self.embeddings)]
        output_emb = torch.cat(output_emb, 2)

        # share embedding layer for both the calendar events
        output_emb_cal = [self.cal_embedding(x_cal_emb[:, :, 0]), self.cal_embedding(x_cal_emb[:, :, 1])]
        output_emb_cal = torch.cat(output_emb_cal, 2)

        x_rnn = torch.cat([x, output_emb, output_emb_cal], 2)

        attn_weights = F.softmax(
            self.attn(torch.cat((x_rnn[0], hidden[0][0], hidden[1][0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.permute(1, 0, 2))

        output = torch.cat((x_rnn[0], attn_applied[:, 0, :]), 1)
        output = self.attn_combine(output).unsqueeze(0).permute(1, 0, 2)

        x_rnn = self.fc(output)
        x_rnn_1 = self.fc_1(x_rnn)
        x_rnn = torch.cat([x_rnn, x_rnn_1], 2).permute(1, 0, 2)
        output, hidden = self.rnn(x_rnn)
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

    def forward(self, x_enc, x_enc_emb, x_cal_enc_emb, x_dec, x_dec_emb, x_cal_dec_emb, x_prev_day_sales_dec):
        batch_size, pred_len = x_dec.shape[0:2]

        # Ignore some initial timesteps of encoder data, according to the max_length allowed
        x_enc, x_enc_emb = x_enc[:, :self.max_length], x_enc_emb[:, :self.max_length]
        x_cal_enc_emb = x_cal_enc_emb[:, :self.max_length]

        # create a tensor to store the outputs
        predictions = torch.zeros(batch_size, pred_len, 9).to(self.config.device)

        encoder_output, hidden = self.encoder(x_enc, x_enc_emb, x_cal_enc_emb)

        # for each prediction timestep, use the output of the previous step,
        # concatenated with other features as the input

        # enable teacher forcing only if model is in training phase
        use_teacher_forcing = True if (random.random() < self.config.teacher_forcing_ratio) & self.training else False

        for timestep in range(0, pred_len):

            if timestep == 0:
                # for the first timestep of decoder, use previous steps' sales
                dec_input = torch.cat([x_dec[:, 0, :], x_prev_day_sales_dec[:, 0]], dim=1).unsqueeze(1)
            else:
                if use_teacher_forcing:
                    dec_input = torch.cat([x_dec[:, timestep, :], x_prev_day_sales_dec[:, timestep]], dim=1).unsqueeze(1)
                else:
                    # for next timestep, current timestep's output will serve as the input along with other features
                    dec_input = torch.cat([x_dec[:, timestep, :], decoder_output[:, 4].unsqueeze(1)], dim=1).unsqueeze(1)

            # the hidden state of the encoder will be the initialize the decoder's hidden state
            decoder_output, hidden, _ = self.decoder(dec_input, x_dec_emb[:, timestep, :].unsqueeze(1),
                                                     x_cal_dec_emb[:, timestep, :].unsqueeze(1), hidden,
                                                     encoder_output)

            # add predictions to predictions tensor
            predictions[:, timestep] = decoder_output

        return predictions


def create_model(config):
    # for item_id, dept_id, cat_id, store_id, state_id respectively
    embedding_sizes = [(3049 + 1, 50), (7 + 1, 4), (3 + 1, 2), (10 + 1, 5), (3 + 1, 2)]
    cal_embedding_sizes = (31, 16)
    num_features_enc = 12 + sum([j for i, j in embedding_sizes]) + cal_embedding_sizes[1] * 2
    num_features_dec = 12 + sum([j for i, j in embedding_sizes]) + cal_embedding_sizes[1] * 2
    enc = Encoder(num_features_enc, embedding_sizes, cal_embedding_sizes, config)
    dec = AttnDecoder(num_features_dec, embedding_sizes, cal_embedding_sizes, 9, config)
    model = Seq2Seq(enc, dec, config)
    model.to(config.device)

    return model
