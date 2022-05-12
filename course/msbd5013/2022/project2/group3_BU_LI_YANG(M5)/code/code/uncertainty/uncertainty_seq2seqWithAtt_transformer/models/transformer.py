import torch
import torch.nn as nn
import torch.utils.data
import math


# Build a seq2seq model
# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_sizes, cal_embedding_sizes, config):
        super(Encoder, self).__init__()
        self.config = config
        self.input_size = input_size
        self.max_length = config.window_length if config.sliding_window \
            else config.training_ts['horizon_start_t'] - config.training_ts['data_start_t']

        self.embeddings = nn.ModuleList([nn.Embedding(classes, hidden_size)
                                         for classes, hidden_size in embedding_sizes])
        self.cal_embedding = nn.Embedding(cal_embedding_sizes[0], cal_embedding_sizes[1])

        self.pos_embedding = nn.Embedding(self.max_length, self.input_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.config.enc_nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.enc_nlayers)

        self.dropout = nn.Dropout(self.config.enc_dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.input_size])).to(self.config.device)

    def forward(self, x, x_emb, x_cal_emb):
        x, x_emb, x_cal_emb = x.permute(1, 0, 2), x_emb.permute(1, 0, 2), x_cal_emb.permute(1, 0, 2)  # make time-major
        output_emb = [emb(x_emb[:, :, i]) for i, emb in enumerate(self.embeddings)]
        output_emb = torch.cat(output_emb, 2)

        # share embedding layer for both the calendar events
        output_emb_cal = [self.cal_embedding(x_cal_emb[:, :, 0]), self.cal_embedding(x_cal_emb[:, :, 1])]
        output_emb_cal = torch.cat(output_emb_cal, 2)

        x = torch.cat([x, output_emb, output_emb_cal], 2).permute(1, 0, 2)

        batch_size = x.shape[0]
        x_len = x.shape[1]

        # Positional Encoding
        pos = torch.arange(0, x_len).unsqueeze(0).repeat(batch_size, 1).to(self.config.device)
        x = self.dropout((x * self.scale) + self.pos_embedding(pos))
        x = x.permute(1, 0, 2)

        # Transformer
        output = self.transformer_encoder(x)

        return output


# Decoder
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_sizes, cal_embedding_sizes, output_size, config):
        super(Decoder, self).__init__()
        self.config = config
        self.input_size = input_size
        self.max_length = 28

        self.embeddings = nn.ModuleList([nn.Embedding(classes, hidden_size)
                                         for classes, hidden_size in embedding_sizes])
        self.cal_embedding = nn.Embedding(cal_embedding_sizes[0], cal_embedding_sizes[1])

        self.pos_embedding = nn.Embedding(self.max_length, self.input_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.input_size, nhead=self.config.dec_nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.dec_nlayers)

        self.dropout = nn.Dropout(self.config.dec_dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.input_size])).to(self.config.device)
        self.pred = nn.Linear(self.input_size, output_size)

    def forward(self, enc_out, y_mask, x, x_emb, x_cal_emb):
        x, x_emb, x_cal_emb = x.permute(1, 0, 2), x_emb.permute(1, 0, 2), x_cal_emb.permute(1, 0, 2)  # make time-major
        output_emb = [emb(x_emb[:, :, i]) for i, emb in enumerate(self.embeddings)]
        output_emb = torch.cat(output_emb, 2)

        # share embedding layer for both the calendar events
        output_emb_cal = [self.cal_embedding(x_cal_emb[:, :, 0]), self.cal_embedding(x_cal_emb[:, :, 1])]
        output_emb_cal = torch.cat(output_emb_cal, 2)

        x = torch.cat([x, output_emb, output_emb_cal], 2).permute(1, 0, 2)

        batch_size = x.shape[0]
        x_len = x.shape[1]

        # Positional Encoding
        pos = torch.arange(0, x_len).unsqueeze(0).repeat(batch_size, 1).to(self.config.device)
        x = self.dropout((x * self.scale) + self.pos_embedding(pos))
        x = x.permute(1, 0, 2)

        # Transformer
        output = self.transformer_decoder(x, enc_out, y_mask)

        output = self.pred(output.permute(1, 0, 2))
        return output


class TransformerModel(nn.Module):
    def __init__(self, encoder, decoder, config):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.pred_len = 28

    def forward(self, x_enc, x_enc_emb, x_cal_enc_emb, x_dec, x_dec_emb, x_cal_dec_emb, x_prev_day_sales_dec):
        batch_size = x_dec.shape[0]

        encoder_output = self.encoder(x_enc, x_enc_emb, x_cal_enc_emb)

        # If running in eval mode, run through the decoder in single steps and use predicted x_prev_sales
        # as actual x_prev_sales is not available
        if self.training:
            x_dec = torch.cat([x_dec, x_prev_day_sales_dec], dim=2)
            y_mask = torch.tril(torch.ones((self.pred_len, self.pred_len))).bool().to(self.config.device)
            decoder_output = self.decoder(encoder_output, y_mask, x_dec, x_dec_emb, x_cal_dec_emb)

        else:
            with torch.no_grad():
                for timestep in range(self.pred_len):
                    if timestep == 0:
                        # for the first timestep of decoder, use previous steps' sales
                        dec_input = torch.cat([x_dec[:, 0, :], x_prev_day_sales_dec[:, 0]], dim=1).unsqueeze(1)
                    else:
                        # for next timestep, current timestep's output will serve as the input along with other features
                        prev_sales = torch.cat([x_prev_day_sales_dec[:, 0].unsqueeze(1),
                                                decoder_output[:, :timestep, 4].view(batch_size, -1, 1)], dim=1)
                        dec_input = torch.cat([x_dec[:, :timestep + 1, :], prev_sales], dim=2)

                    y_mask = torch.tril(torch.ones((timestep + 1, timestep + 1))).bool().to(self.config.device)
                    decoder_output = self.decoder(encoder_output, y_mask, dec_input,
                                                  x_dec_emb[:, :timestep + 1, :].view(batch_size, timestep + 1, -1),
                                                  x_cal_dec_emb[:, :timestep + 1, :]
                                                  .view(batch_size, timestep + 1, -1))

        return decoder_output


def create_model(config):
    # for item_id, dept_id, cat_id, store_id, state_id respectively
    embedding_sizes = [(3049 + 1, 50), (7 + 1, 4), (3 + 1, 2), (10 + 1, 5), (3 + 1, 2)]
    cal_embedding_sizes = (31, 16)
    num_features_enc = 12 + sum([j for i, j in embedding_sizes]) + cal_embedding_sizes[1] * 2
    num_features_dec = 12 + sum([j for i, j in embedding_sizes]) + cal_embedding_sizes[1] * 2

    # adjust embed_dims to be divisible by num_heads
    add_embeds = (math.ceil(num_features_enc/config.enc_nhead) * config.enc_nhead) - num_features_enc
    num_features_enc += add_embeds
    num_features_dec += add_embeds
    embedding_sizes[0] = (embedding_sizes[0][0], embedding_sizes[0][1] + add_embeds)

    enc = Encoder(num_features_enc, embedding_sizes, cal_embedding_sizes, config)
    dec = Decoder(num_features_dec, embedding_sizes, cal_embedding_sizes, 9, config)
    model = TransformerModel(enc, dec, config)
    model.to(config.device)

    return model
