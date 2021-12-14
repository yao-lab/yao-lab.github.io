# Here you can
# 1. import necessary python packages for your strategy
# 2. Load your own facility files containing functions, trained models, extra data, etc for later use
# 3. Set some global constants
# Note:
# 1. You should put your facility files in the same folder as this strategy.py file
# 2. When load files, ALWAYS use relative path such as "data/facility.pickle"
# DO NOT use absolute path such as "C:/Users/Peter/Documents/project/data/facility.pickle"
# asset_index = 1  # only consider BTC (the **second** crypto currency in dataset)

# Here is your main strategy function
# Note:
# 1. DO NOT modify the function parameters (time, data, etc.)
# 2. The strategy function AWAYS returns two things - position and memory:
# 2.1 position is a np.array (length 4) indicating your desired position of four crypto currencies next minute
# 2.2 memory is a class containing the information you want to save currently for future use
import numpy as np

BAR_LENGTH = 720
SPLIT_THRESHOLD = 0.5
ASSETS_MAPPING = {'BTCUSDT': 0, 'ETHUSDT': 1, 'LTCUSDT': 2, 'XRPUSDT': 3}
CASH_WEIGHT = 0.1

def handle_bar(counter,  # a counter for number of minute bars that have already been tested
               time,  # current time in string format such as "2018-07-30 00:30:00"
               data,  # data for current minute bar (in format 2)
               init_cash,  # your initial cash, a constant
               transaction,  # transaction ratio, a constant
               cash_balance,  # your cash balance at current minute
               crypto_balance,  # your crpyto currency balance at current minute
               total_balance,  # your total balance at current minute
               position_current,  # your position for 4 crypto currencies at this minute
               memory # a class, containing the information you saved so far
               ):
    # Here you should explain the idea of your strategy briefly in the form of Python comment.
    # You can also attach facility files such as text & image & table in your team folder to illustrate your idea

    # The idea of my strategy:
    # Save every 720 minute bars to produce signals
    position = position_current  # load current position
    if ((counter + 1) % BAR_LENGTH == 0):
        factor_dict = get_signal(memory)
        position = calcultae_target_position(factor_dict, cash_balance, position, data, CASH_WEIGHT)
        # clean the memory data
        memory.clean_memory()
    else:
        memory.save_data(counter, data)
    return position, memory

def get_signal(memory):
    factor_dict = {}
    for asset, hist_data in memory.data_save.items():
        hist_data = hist_data.sort_values(by='volume')
        # Use the average volume as the threshold
        vol_threshold = hist_data['volume'].mean()
        # split the data into momentum part and the reversal part based on the threshold
        mom_data = hist_data[hist_data['volume'] <= vol_threshold]
        rev_data = hist_data[hist_data['volume'] > vol_threshold]
        # calulate the factor for each asset
        mom_factor = calculate_factor(mom_data, 'momentum')
        rev_factor = calculate_factor(rev_data, 'reversal')
        factor = rev_factor - mom_factor
        factor_dict[asset] = factor
    return factor_dict

def calcultae_target_position(factor_dict, cash_balance, position, data, cash_weight):
    # calculate the target position based on the signals
    threshold = 0.001
    for asset, factor in factor_dict.items():
        if factor > 0 + threshold:
            if cash_balance >= 5000:
                weight = factor / sum([k for k in list(factor_dict.values()) if k > 0])
                position[ASSETS_MAPPING[asset]] = weight * cash_balance * cash_weight / data[ASSETS_MAPPING[asset]][0]
        elif factor < 0 - threshold:
            weight = factor / sum([k for k in list(factor_dict.values()) if k < 0])
            position[ASSETS_MAPPING[asset]] = -(weight * cash_balance * cash_weight / data[ASSETS_MAPPING[asset]][0])
    return position

def calculate_factor(data, method):
    # weight is reverse proportional to the volume
    if method == 'momentum':
        weighted_list = (1 / data['volume']) / ((1 / data['volume']).sum())
    # weight is directly proportional to the volume
    elif method == 'reversal':
        weighted_list = data['volume'] / (data['volume'].sum())
    factor =  (weighted_list * np.log(data['open'] / (data['open'].shift(1)))).sum()
    return factor