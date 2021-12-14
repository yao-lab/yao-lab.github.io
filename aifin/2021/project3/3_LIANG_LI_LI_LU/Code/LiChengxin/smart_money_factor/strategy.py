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

BAR_LENGTH = 600
SPLIT_THRESHOLD = 0.5
ASSETS_MAPPING = {'BTCUSDT': 0, 'ETHUSDT': 1, 'LTCUSDT': 2, 'XRPUSDT': 3}
BETA_LIST = [-0.5, -0.25, -0.1, 0, 0.05, 0.1, 0.25, 0.33, 0.5, 0.7, 1]


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
    position = position_current  # load current position
    # Save every 600 minute bars to produce signals
    if ((counter + 1) % BAR_LENGTH == 0):
        indicator_dict = calculate_indicator(memory)
        factor_dict = calculate_factor(indicator_dict)
        position = calcultae_target_position(factor_dict, cash_balance, position, data)
        # clean the memory data
        memory.clean_memory()
    else:
        memory.save_data(counter, data)
    return position, memory


def calcultae_target_position(factor_dict, cash_balance, position, data):
    # for each asset, calculate the target position based on the signals
    threshold = 0.001
    for asset, factor in factor_dict.items():
        if factor > 1 + threshold:
            if cash_balance >= 10000:
                weight = factor / sum([k for k in list(factor_dict.values()) if k > 1])
                position[ASSETS_MAPPING[asset]] = weight * cash_balance * 0.25 / data[ASSETS_MAPPING[asset]][3]
        elif factor < 1 - threshold:
            weight = factor / sum([k for k in list(factor_dict.values()) if k < 1])
            position[ASSETS_MAPPING[asset]] = -(weight * cash_balance * 0.25 / data[ASSETS_MAPPING[asset]][3])
    return position


def calculate_factor(indicator_dict):
    # For each asset, calculte the factor list (different beta)
    factor_record = {}
    for asset, data in indicator_dict.items():
        vwap_all = ((data['close'] * data['volume']).sum()) / data['volume'].sum()
        factor_dict = {}
        for beta in BETA_LIST:
            data_beta = data.sort_values(by='s_indicator_{}'.format(str(beta)))
            data_beta['cumsum_vol'] = data_beta['volume'].cumsum() / data_beta['volume'].sum()
            smart_data = data_beta[data_beta['cumsum_vol'] <= 0.2].reset_index(drop=True)
            vwap_smart = ((smart_data['close'] * smart_data['volume']).sum()) / smart_data['volume'].sum()
            factor_dict['Q_factor_{}'.format(str(beta))] = vwap_smart / vwap_all
        # volume Q
        data_volume = data.sort_values(by='s_indicator_volume')
        data_volume['cumsum_vol'] = data_volume['volume'].cumsum() / data_volume['volume'].sum()
        smart_data = data_volume[data_volume['cumsum_vol'] <= 0.2].reset_index(drop=True)
        vwap_smart_volume = ((smart_data['close'] * smart_data['volume']).sum()) / smart_data['volume'].sum()
        Q_factor_volume = vwap_smart_volume / vwap_all
        # volume rank
        data_rank = data.sort_values(by='s_indicator_sum_rank')
        data_rank['cumsum_vol'] = data_rank['volume'].cumsum() / data_rank['volume'].sum()
        smart_data = data_rank[data_rank['cumsum_vol'] <= 0.2].reset_index(drop=True)
        vwap_smart_rank = ((smart_data['close'] * smart_data['volume']).sum()) / smart_data['volume'].sum()
        Q_factor_rank = vwap_smart_rank / vwap_all
        # volume ln volume
        data_ln = data.sort_values(by='s_indicator_ln_volume')
        data_ln['cumsum_vol'] = data_ln['volume'].cumsum() / data_ln['volume'].sum()
        smart_data = data_ln[data_ln['cumsum_vol'] <= 0.2].reset_index(drop=True)
        vwap_smart_ln = ((smart_data['close'] * smart_data['volume']).sum()) / smart_data['volume'].sum()
        Q_factor_ln = vwap_smart_ln / vwap_all

        factor_list = [factor_dict['Q_factor_-0.5'], factor_dict['Q_factor_-0.25'], factor_dict['Q_factor_-0.1'],
                       factor_dict['Q_factor_0'],
                       factor_dict['Q_factor_0.05'], factor_dict['Q_factor_0.1'], factor_dict['Q_factor_0.25'],
                       factor_dict['Q_factor_0.33'],
                       factor_dict['Q_factor_0.5'], factor_dict['Q_factor_0.7'], factor_dict['Q_factor_1'],
                       Q_factor_volume, Q_factor_rank,
                       Q_factor_ln]
        ##########################################
        factor_record[asset] = factor_list[1]
    return factor_record


def calculate_indicator(memory):
    # for each asset, calculate the indicator S
    factor_dict = {}
    for asset, data in memory.data_save.items():
        data['change'] = (data['close'] - data['open']) / data['open']
        for beta in BETA_LIST:
            data['s_indicator_{}'.format(str(beta))] = abs(data['change']) / (np.power(data['volume'], beta))
        data['s_indicator_volume'] = data['volume']
        data['s_indicator_volume_rank'] = data['volume'].rank(method='dense')
        data['s_indicator_change_rank'] = data['change'].rank(method='dense')
        data['s_indicator_sum_rank'] = data['s_indicator_volume_rank'] + data['s_indicator_change_rank']
        data['s_indicator_ln_volume'] = abs(data['change']) / np.log(data['volume'])
        factor_dict[asset] = data
    return factor_dict
