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

BAR_LENGTH = 30
ASSETS_MAPPING = {'BTCUSDT': 0, 'ETHUSDT': 1, 'LTCUSDT': 2, 'XRPUSDT': 3}


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
    # The idea of my strategy:
    # We calculated the VRI factors as below,
    # we buy the asset if the VRI is smaller than the lower bound and sell the position if the VRI is higher than the upper bound.
    # For lower and upper bounds, we set 0.01 and 0.95 quantile of VRI of the last thirty minutes.
    # We use open price minus close price as the difference term,
    # then take the difference between the smallest low price and largest high price of last three bars as ExtremeRange term.
    # The volatility is calculated by the standard deviation of the close price with five rolling windows.

    position = position_current  # load current position
    memory.save_data(counter, data)
    factor_dict, memory = get_threshold(counter, memory)
    position = calcultae_target_position(factor_dict, cash_balance, position, data)

    return position, memory


# every 30 mins 
def get_threshold(counter, memory):
    factor_dict = {}
    for asset, hist_data in memory.data_save.items():
        if counter > BAR_LENGTH:
            hist_data['vol'] = hist_data.rolling(window=5)['close'].std()
            hist_data['min'] = hist_data.rolling(window=5)['low'].min()
            hist_data['max'] = hist_data.rolling(window=5)['high'].max()
            hist_data['priceChange'] = hist_data['open'] - hist_data['close']
            hist_data['VRI'] = (hist_data['priceChange'] / (hist_data['min'] - hist_data['max'])) * hist_data['vol']
            if ((counter - 1) % BAR_LENGTH == 0):
                memory.get_threshold(asset, hist_data['VRI'].tail(BAR_LENGTH))

            temp_threshold_dict = memory.threshold_dict[asset]
            weight = hist_data['volume'].iloc[-1] / hist_data['volume'].sum()
            # long
            if (hist_data['VRI'].iloc[-1] >= temp_threshold_dict.iloc[0]) & (
                    hist_data['VRI'].iloc[-2] < temp_threshold_dict.iloc[0]):
                factor_dict[asset] = 1
            if (hist_data['VRI'].iloc[-1] <= temp_threshold_dict.iloc[1]) & (
                    hist_data['VRI'].iloc[-2] > temp_threshold_dict.iloc[1]):
                factor_dict[asset] = -1

    return factor_dict, memory


def calcultae_target_position(factor_dict, cash_balance, position, data):
    threshold = 0.001
    for asset, factor in factor_dict.items():
        if factor > 0 + threshold:
            if cash_balance >= 20000:
                weight = factor / 4
                position[ASSETS_MAPPING[asset]] = min(weight * cash_balance * 0.3 / data[ASSETS_MAPPING[asset]][3],
                                                      30000 / data[ASSETS_MAPPING[asset]][3])
        elif factor < 0 - threshold:
            weight = factor / 4
            position[ASSETS_MAPPING[asset]] = -(weight * cash_balance * 0.3 / data[ASSETS_MAPPING[asset]][3])
    return position