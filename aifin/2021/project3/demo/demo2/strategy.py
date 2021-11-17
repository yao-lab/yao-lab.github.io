# Here you can
# 1. import necessary python packages for your strategy
# 2. Load your own facility files containing functions, trained models, extra data, etc for later use
# 3. Set some global constants
# Note:
# 1. You should put your facility files in the same folder as this strategy.py file
# 2. When load files, ALWAYS use relative path such as "data/facility.pickle"
# DO NOT use absolute path such as "C:/Users/Peter/Documents/project/data/facility.pickle"
import numpy as np
from auxiliary import generate_bar, white_soider, black_craw  # auxiliary is a local py file containing some functions

bar_length = 15  # Number of minutes to generate next new bar
asset_index = 1  # only consider BTC (the **second** crypto currency in dataset)
my_cash_balance_lower_limit = 30000.  # Cutting-loss criterion

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
    # Make use of some technical patterns in candlestick plots:

    # Pattern for long signal: (unstrict) three white soldiers
    # For three sequential bars (each bar is 15 minutes long), if they have the pattern "white soldiers", i.e.
    # 1. OPEN(n) > OPEN(n-1)
    # 2. CLOSE(n) > CLOSE(n-1)
    # 3. CLOSE(n) > OPEN(n)
    # then we long 1 BTC at next bar unless the current cash balance is less than 30,000
    # stop loss point: When the price drop down to the close price of the first white soilder, clear all long position
    # target profit point: When the price go up to (1+5%) times the close price of the third white soilder, clear all long position

    # Pattern for short signal: (unstrict) three black craws
    # For three sequential bars (each bar is 15 minutes long), if they have the pattern "black craws", i.e.
    # 1. OPEN(n) < OPEN(n-1)
    # 2. CLOSE(n) < CLOSE(n-1)
    # 3. CLOSE(n) < OPEN(n)
    # then we short 1 BTC at next bar unless the current cash balance is less than 30,000
    # stop loss point: When the price go up to the close price of the first black craw, clear all short position
    # target profit point: When the price go up to (1-5%) times the close price of the third black craw, clear all long position

    # Get position of last minute
    position_new = position_current
    
    # Generate OHLC data for every 15 minutes
    if counter == 0:
        memory.data_list = list()
        memory.bar_prev = np.array([None])
        memory.ws_check_table = np.empty((0,2))
        memory.bc_check_table = np.empty((0,2))
        memory.long_stop_loss = np.inf
        memory.long_profit_target = np.inf
        memory.short_stop_loss = np.inf
        memory.short_profit_target = np.inf
    
    if (counter + 1) % bar_length == 0:
        memory.data_list.append(data)
        bar = generate_bar(memory.data_list)
        memory.data_list = list()  # Clear memory.data_list after bar combination
        
        if memory.bar_prev.any()!=None:
            ws_check = white_soider(bar, memory.bar_prev, asset_index)
            memory.ws_check_table = np.append(memory.ws_check_table, [ws_check], axis=0)
            bc_check = black_craw(bar, memory.bar_prev, asset_index)
            memory.bc_check_table = np.append(memory.bc_check_table, [bc_check], axis=0)
            
        bar_num = len(memory.ws_check_table)
        if bar_num>3:
            ''' long signal 
                When there is a three white soider signal, long 1 BTC at next minute unless 
                the current cash balance is less than my_cash_balance_lower_limit
            '''
            if np.sum(memory.ws_check_table[(bar_num-3):bar_num,1])==3:
                if cash_balance > my_cash_balance_lower_limit:
                    position_new[asset_index] += 1.
                    memory.long_stop_loss = memory.ws_check_table[bar_num-3,0]
                    memory.long_profit_target = memory.ws_check_table[bar_num-1,0]*(1+.05)
            
            ''' short signal 
                When there is a three black craw signal, short 1 BTC at next minute unless 
                the current cash balance is less than my_cash_balance_lower_limit
            '''
            if np.sum(memory.bc_check_table[(bar_num-3):bar_num,1])==3:
                if cash_balance > my_cash_balance_lower_limit:
                    position_new[asset_index] -= 1.
                    memory.short_stop_loss = memory.bc_check_table[bar_num-3,0]
                    memory.short_profit_target = memory.bc_check_table[bar_num-1,0]*(1-.05)
        
        memory.bar_prev = bar
        
    # save minute data to data_list
    else:
        memory.data_list.append(data)
    
    # Close signal
    # When reach stop loss/target profit points, clear all long/short positions
    average_price = np.mean(data[asset_index,:4])
    if(position_new[asset_index] > 0):
        if average_price > memory.long_profit_target or average_price < memory.long_stop_loss:
            position_new[asset_index] = 0.
    else:
        if average_price > memory.short_stop_loss or average_price < memory.short_profit_target:
            position_new[asset_index] = 0.
    # End of strategy
    return position_new, memory
