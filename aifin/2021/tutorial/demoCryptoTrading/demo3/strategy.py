# Here you can
# 1. import necessary python packages for your strategy
# 2. Load your own facility files containing functions, trained models, extra data, etc for later use
# 3. Set some global constants
# Note:
# 1. You should put your facility files in the same folder as this strategy.py file
# 2. When load files, ALWAYS use relative path such as "data/facility.pickle"
# DO NOT use absolute path such as "C:/Users/Peter/Documents/project/data/facility.pickle"
from auxiliary import generate_bar
import pandas as pd
from sklearn.externals import joblib

model = joblib.load('model.pkl')#####
asset_index = 1  # only consider BTC (the **second** crypto currency in dataset)
bar_length = 30  # Number of minutes to generate next new bar

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
    # Logistic regression with label = rising/falling signal. 

    # Pattern for long signal:
    # When the predicted signal is rising, we long 1 BTC at the next bar; otherwise we short 1 BTC at the next bar.

    # Pattern for short signal:
    # When the predicted probability of rising is low (i.e., lower than 0.45), we short 1 BTC at the next bar.

    # No controlling of the position is conducted in this strategy.

    # Get position of last minute
    position_new = position_current
    
    # Generate OHLC data for every 30 minutes
    if (counter == 0):
        #memory.data_save = np.zeros((bar_length, 5))#, dtype=np.float64)
        memory.data_save = pd.DataFrame(columns = ['close', 'high', 'low', 'open', 'volume'])

    if ((counter + 1) % bar_length == 0):
        memory.data_save.loc[bar_length - 1] = data[asset_index,]
        bar = generate_bar(memory.data_save) # pandas dataframe
        bar_X = bar[['open', 'close']]

        prob_pred = model.predict_proba(bar_X)[:,1]
        if (prob_pred > 0.55): position_new[asset_index] += 1
        if (prob_pred < 0.45): position_new[asset_index] -= 1
    else:
        memory.data_save.loc[(counter + 1) % bar_length - 1] = data[asset_index,]#####

    # End of strategy
    return position_new, memory
