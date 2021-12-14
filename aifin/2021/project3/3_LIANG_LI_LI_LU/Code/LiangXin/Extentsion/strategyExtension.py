# Here you can
# 1. import necessary python packages for your strategy
# 2. Load your own facility files containing functions, trained models, extra data, etc for later use
# 3. Set some global constants
# Note:
# 1. You should put your facility files in the same folder as this strategy.py file
# 2. When load files, ALWAYS use relative path such as "data/facility.pickle"

# Here is your main strategy function
# Note:
# 1. DO NOT modify the function parameters (time, data, etc.)
# 2. The strategy function AWAYS returns two things - position and memory:
# 2.1 position is a np.array (length 4) indicating your desired position of four crypto currencies next minute
# 2.2 memory is a class containing the information you want to save currently for future use

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

BATCH_SIZE = 150
HIDDEN_SIZE = 50
DROPOUT = 0.1
NUM_LAYERS = 3

OUTPUT_SIZE = 1
NUM_FEATURES = 5

params = {'batch_size': BATCH_SIZE,
          'shuffle': False,
          'drop_last': True,
          'num_workers': 4}

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, directions=1):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.directions = directions

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, output_size)

    def init_hidden_states(self, batch_size):
        state_dim = (self.num_layers * self.directions, batch_size, self.hidden_size)
        return (torch.zeros(state_dim), torch.zeros(state_dim))

    def forward(self, x, states):
        x, (h, c) = self.lstm(x, states)
        out = self.linear(x)
        return out, (h, c)

seed = 32
torch.manual_seed(seed)

model_load = LSTM(NUM_FEATURES, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT)
model_load.load_state_dict(torch.load(os.getcwd() + '/LiangXin/LSTM/model.pt', map_location='cpu'))

def generate_bar(data_list):

    ''' Function to reform format2 data into the time period you want
    Param: data_list - list containing n sequential elements from data_format2
    '''

    open_price = data_list[0][:, 3]
    high = np.max([array[:, 1] for array in data_list], axis=0)
    low = np.min([array[:, 2] for array in data_list], axis=0)
    close_price = data_list[-1][:, 0]
    volume = np.sum([array[:, 4] for array in data_list], axis=0)
    OHLC = np.array([open_price, high, low, close_price, volume]).T
    return OHLC


bar_length = 60*9

def handle_bar(counter,  # a counter for number of minute bars that have already been tested
               time,  # current time in string format such as "2018-07-30 00:30:00"
               data,  # data for current minute bar (in format 2)
               init_cash,  # your initial cash, a constant
               transaction,  # transaction ratio, a constant
               cash_balance,  # your cash balance at current minute
               crypto_balance,  # your crpyto currency balance at current minute
               total_balance,  # your total balance at current minute
               position_current,  # your position for 4 crypto currencies at this minute
               memory  # a class, containing the information you saved so far
               ):
    # Here you should explain the idea of your strategy briefly in the form of Python comment.
    # You can also attach facility files such as text & image & table in your team folder to illustrate your idea

    # The idea of my strategy:
    # When the time arrives each 540 minutes, we put the 540-minute k-line data including open, high, low, close price,
    # and combined volume for every crypto currency into trained LSTM model. Higher predicted return from the model,
    # more investment capital to the underlying asset limited to $20,000 in total for each time. However, as the cash
    # balance is lower than $20,000, we only short the asset with negative predicted return due to the cash limit by
    # the rule.

    # Get position of last minute
    position_new = position_current

    # Generate OHLC data for every 60*9 minutes
    if counter == 0:
       memory.data_list = list()

    elif (counter+1) % bar_length == 0:
        memory.data_list.append(data)
        bar = generate_bar(memory.data_list)
        memory.data_list = list()

        states = model_load.init_hidden_states(1)
        torch_data = (torch.from_numpy(bar)).unsqueeze(0)
        output, _ = model_load(torch_data.to(torch.float32), states)

        if cash_balance <= 20000:
            id = np.where((output[0] < 0).reshape(1, -1)[0])[0]
            weight = -(pd.Series((output.detach().numpy()[0].reshape(1, -1)[0])[id]).rank(ascending=False)/(pd.Series((output.detach().numpy()[0].reshape(1, -1)[0])[id]).rank(ascending=False)).sum()).values
            # investmentWeights = np.zeros(4)
            investmentWeights = np.zeros(3)
            investmentWeights[id] = weight
            output01 = np.sign(output.detach().numpy()).reshape(1, -1) * np.min(np.array([(cash_balance-11000), 20000])) * investmentWeights / data[:, 3]
            position_new += output01[0]

        else:
            weight = (pd.Series(output.detach().numpy()[0].reshape(1, -1)[0]).abs().rank() / pd.Series(output.detach().numpy()[0].reshape(1, -1)[0]).abs().rank().sum()).sort_index().values.reshape(-1, 1)
            output01 = np.sign(output.detach().numpy()) * 20000 * weight / data[:, 3].reshape(-1, 1)
            position_new += output01[0].T[0]

    else:
        memory.data_list.append(data)

    return position_new, memory


pnl = pd.read_csv('/Users/liangxin/Desktop/MAFS6010Z/Project3/backtest_details.csv')
