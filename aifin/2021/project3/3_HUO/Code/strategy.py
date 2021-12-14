# Here you can
# 1. import necessary python packages for your strategy
# 2. Load your own facility files containing functions, trained models, extra data, etc for later use
# 3. Set some global constants
# Note:
# 1. You should put your facility files in the same folder as this strategy.py file
# 2. When load files, ALWAYS use relative path such as "data/facility.pickle"
# DO NOT use absolute path such as "C:/Users/Peter/Documents/project/data/facility.pickle"
import torch
from torch import nn
import pandas as pd
import numpy as np
from auxiliary import generate_X
import DNN

asset_index = 0  # only consider BTC (the **second** crypto currency in dataset)

dnn_model = torch.load('dnn.mdl')
bar_length = 6
# Here is your main strategy function
# Note:
# 1. DO NOT modify the function parameters (time, data, etc.)
# 2. The strategy function AWAYS returns two things - position and memory:
# 2.1 position is a np.array (length 4) indicating your desired position of four crypto currencies next minute
# 2.2 memory is a class containing the information you want to save currently for future use


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
    # Buy 10 BTC at the very beginning and hold it to the end.
    position_new = np.repeat(0., 1)  # load current position
    if (counter == 0):
        memory.data_save = pd.DataFrame(columns = ['Open', 'High', 'Low', 'Close', 'Volume'])
    if (counter <= 5):
        memory.data_save.loc[counter] = data[asset_index,]
    else:
        memory.data_save.shift(-1)
        memory.data_save.loc[bar_length - 1] = data[asset_index,]#####

    #if ((counter + 1) % bar_length == 0):
    memory.data_save.loc[bar_length - 1] = data[asset_index,]
    X = generate_X(memory.data_save) # pandas dataframe
    #bar_X = bar[['open', 'close']]
    X = torch.FloatTensor(X)
    prob = nn.Softmax(dim = 1)
    p = prob(dnn_model.forward(X)).detach().numpy()

    for i in range(p.shape[0]):
        if np.max(p[i, 1]) >= 0.95:
            position_new[asset_index] = 1 * total_balance/data[asset_index,3]/2
        else:
            position_new[asset_index] = 0

        #prob_pred = dnn_model.predict(bar)[:,1]
        #if (prob_pred > 0.55): position_new[asset_index] += 1
        #if (prob_pred < 0.45): position_new[asset_index] -= 1
    # else:
    #     #print(data[asset_index,])
    #     memory.data_save.loc[(counter + 1) % bar_length - 1] = data[asset_index,]#####

    # End of strategy
    return position_new, memory