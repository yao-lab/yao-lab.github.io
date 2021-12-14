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

import numpy as np


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
    # When the short-term EMA crosses the long-term EMA from bottom to top, the long signal is generated.
    # Otherwise, when the short-term EMA crosses the long-term EMA from top to bottom, the short signal is generated.
    # We capture the signal in every 30 minutes, and set the alpha as 0.7.
    # Each currency is traded with 2500 dollars per time, when cash balance is less than 20000 dollars, the position will
    # be automatically cleared, and wait for the next signal.

    # Get position of last minute
    position_new = position_current
    alpha = 0.7
    output = np.repeat(0., 4)

    # Generate OHLC data for every 15 minutes
    if counter == 0:
        memory.data_list = list()
        memory.EMA5 = list()
        memory.EMA20 = list()
        memory.EMA5v = np.repeat(0., 4)
        memory.EMA20v = np.repeat(0., 4)
        memory.ifgold = np.repeat(0., 4)
        memory.sumifgold = np.repeat(0., 4)
        memory.first = 0

    elif counter <= 5:
        memory.EMA5.append(data[:, 0])
        memory.EMA20.append(data[:, 0])
    elif counter <= 20:
        memory.EMA20.append(data[:, 0])

        del (memory.EMA5[0])
        memory.EMA5.append(data[:, 0])
    else:
        del (memory.EMA20[0])
        memory.EMA20.append(data[:, 0])
        del (memory.EMA5[0])
        memory.EMA5.append(data[:, 0])

        memory.EMA5v = memory.EMA5[0][:]
        for i in range(len(memory.EMA5)):
            memory.EMA5v = alpha * memory.EMA5[i][:] + (1 - alpha) * memory.EMA5v

        memory.EMA20v = memory.EMA20[0][:]
        for i in range(len(memory.EMA20)):
            memory.EMA20v = alpha * memory.EMA20[i][:] + (1 - alpha) * memory.EMA20v

        ifgold_new = (memory.EMA5v > memory.EMA20v) + 0

        if memory.first == 1:
            output = ifgold_new - memory.ifgold
        memory.ifgold = ifgold_new
        memory.first = 1
        memory.sumifgold += memory.ifgold

        if (counter + 1) % 30 == 0:

            if cash_balance <= 20000:
                position_new = np.repeat(0., 4)
            else:
                weight = np.repeat(0.25, 4)
                output = output * 10000 * weight / data[:, 0].reshape(1, -1)
                position_new += output[0]
            memory.sumifgold = np.repeat(0., 4)

    return position_new, memory