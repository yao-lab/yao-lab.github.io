import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame


def generate_X(data1):
    
    data = DataFrame.copy(data1, deep=True)
    data['Open'] = data['Open'].apply(np.log)
    data['High'] = data['High'].apply(np.log)
    data['Low'] = data['Low'].apply(np.log)
    data['Close'] = data['Close'].apply(np.log)

    data['H_O'] = (data['High'] - data['Open'])
    data['C_O'] = (data['Close'] - data['Open'])
    data['C_L'] = (data['Close'] - data['Low'])
    data['H_L'] = (data['High'] - data['Low'])

    data['Volume'] = data['Volume']

    data['vol1'] = data['Volume'].shift(1)
    data['vol2'] = data['Volume'].shift(2)
    data['vol3'] = data['Volume'].shift(3)
    data['vol4'] = data['Volume'].shift(4)
    data['vol5'] = data['Volume'].shift(5)

    data['vol1'] = data['Volume']/data['vol1']
    data['vol2'] = data['Volume']/data['vol2']
    data['vol3'] = data['Volume']/data['vol3']
    data['vol4'] = data['Volume']/data['vol4']
    data['vol5'] = data['Volume']/data['vol5']

    data['C_O1'] = data['C_O'].shift(1)
    data['C_O2'] = data['C_O'].shift(2)
    data['C_O3'] = data['C_O'].shift(3)
    data['C_O4'] = data['C_O'].shift(4)
    data['C_O5'] = data['C_O'].shift(5)

    #print(data)
    data.dropna(inplace=True)
    #print(data)
    #data = data.drop(columns=['Time UTC', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades'])
    data = data.drop(columns=['Volume'])
    return np.array(data)