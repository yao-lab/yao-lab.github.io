# modified from source: https://github.com/zalandoresearch/pytorch-dilated-rnn/blob/master/drnn.py
# to extract outputs from each layer and both hidden and cell state for LSTM

import torch
import torch.nn as nn


use_cuda = torch.cuda.is_available()


class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dropout=0, cell_type='GRU', batch_first=False, layer_outputs=False):
        super(DRNN, self).__init__()

        self.dilations = [2 ** i for i in range(n_layers)]
        self.cell_type = cell_type
        self.batch_first = batch_first
        self.layer_outputs = layer_outputs

        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        self.rnn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(n_layers - 1)])

        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden)
            else:
                c = cell(n_hidden, n_hidden)
            layers.append(c)
        self.cells = nn.Sequential(*layers)

    def forward(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        output_hiddens, output_seqs, output_hiddens_lstm = [], [], [[], []]
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if i != 0:
                inputs = self.rnn_dropouts[i - 1](inputs)
            if hidden is None:
                inputs, hid = self.drnn_layer(cell, inputs, dilation)
                if self.cell_type == 'LSTM':
                    output_hiddens_lstm[0].append(hid[0])
                    output_hiddens_lstm[1].append(hid[1])
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])
                if self.cell_type == 'LSTM':
                    output_hiddens_lstm[0].append(hidden[i][0])
                    output_hiddens_lstm[1].append(hidden[i][1])

            output_hiddens.append(inputs[-dilation:])
            output_seqs.append(inputs)

        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        if self.cell_type == 'LSTM':
            output_hiddens = output_hiddens_lstm

        if self.layer_outputs:
            return inputs, output_hiddens, output_seqs
        else:
            return inputs, output_hiddens

    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, _ = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        is_even = (n_steps % rate) == 0

        if not is_even:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            if use_cuda:
                zeros_ = zeros_.cuda()

            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim)
        if use_cuda:
            hidden = hidden.cuda()
        if self.cell_type == "LSTM":
            memory = torch.zeros(batch_size, hidden_dim)
            if use_cuda:
                memory = memory.cuda()
            return (hidden, memory)
        else:
            return hidden
