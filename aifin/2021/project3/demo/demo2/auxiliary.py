import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_bar(data_list):
    ''' Function to reform format2 data into the time period you want
    Param: data_list - list containing n sequential elements from data_format2
    '''
    import numpy as np
    open_price = data_list[0][:, 3]
    high = np.max([array[:, 1] for array in data_list],axis=0)
    low = np.min([array[:, 2] for array in data_list],axis=0)
    close_price = data_list[-1][:, 0]
    OHLC = np.array([open_price,high,low,close_price]).T
    
    return OHLC


def white_soider(data_cur, data_prev, asset=1):
    '''
    Params: data_cur - current minute data matrix
            data_prev - previous minute dta matrix
            asset - index of asset, here we use BTC as default
    '''
    open_cur = data_cur[asset,0]
    close_cur = data_cur[asset,3]
    open_prev = data_prev[asset,0]
    close_prev = data_prev[asset,3]
    is_white_soider = (open_cur > open_prev) and (close_cur > close_prev) and (close_cur > open_cur)
    
    return [close_cur, is_white_soider]


def black_craw(data_cur, data_prev, asset=1):
    '''
    Params: data_cur - current minute data matrix
            data_prev - previous minute dta matrix
            asset - index of asset, here we use BTC as default
    '''
    open_cur = data_cur[asset,0]
    close_cur = data_cur[asset,3]
    open_prev = data_prev[asset,0]
    close_prev = data_prev[asset,3]
    is_black_craw = (open_cur < open_prev) and (close_cur < close_prev) and (close_cur < open_cur)
    
    return [close_cur, is_black_craw]

def plot_candles(pricing, fig_length=18, fig_height=10, title=None, volume_bars=False, color_function=None, technicals=None):
    """ Plots a candlestick chart using quantopian pricing data.
    Args:
      pricing: A pandas dataframe with columns ['open_price', 'close_price', 'high', 'low', 'volume']
      title: An optional title for the chart
      volume_bars: If True, plots volume bars
      color_function: A function which, given a row index and price series, returns a candle color.
      technicals: A list of additional data series to add to the chart.  Must be the same length as pricing.
    """
    def default_color(index, open_price, close_price, low, high):
        return 'r' if open_price[index] > close_price[index] else 'g'
    color_function = color_function or default_color
    technicals = technicals or []
    open_price = pricing['open']
    close_price = pricing['close']
    low = pricing['low']
    high = pricing['high']
    oc_min = pd.concat([open_price, close_price], axis=1).min(axis=1)
    oc_max = pd.concat([open_price, close_price], axis=1).max(axis=1)
    
    if volume_bars:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3,1]})
    else:
        fig, ax1 = plt.subplots(1, 1)
        
    # Set figure size
    fig.set_size_inches(fig_length, fig_height)
    
    if title:
        ax1.set_title(title)
    x = np.arange(len(pricing))
    candle_colors = [color_function(i, open_price, close_price, low, high) for i in x]
    candles = ax1.bar(x, oc_max-oc_min, bottom=oc_min, color=candle_colors, linewidth=0)
    lines = ax1.vlines(x + 0.4, low, high, color=candle_colors, linewidth=1)
    ax1.xaxis.grid(False)
    ax1.xaxis.set_tick_params(which='major', length=3.0, direction='in', top='off')
    
    # Assume minute frequency if first two bars are in the same day
    frequency = 'minute' if (pricing.index[1] - pricing.index[0]).days == 0 else 'day'
    time_format = '%d-%m-%Y'
    if frequency == 'minute':
        time_format = '%H:%M'
        
    # Set X axis tick labels
    plt.xticks(x, [date.strftime(time_format) for date in pricing.index], rotation='vertical')
    for indicator in technicals:
        ax1.plot(x, indicator)
    
    if volume_bars:
        volume = pricing['volume']
        volume_scale = None
        scaled_volume = volume
        if volume.max() > 1000000:
            volume_scale = 'M'
            scaled_volume = volume / 1000000
        elif volume.max() > 1000:
            volume_scale = 'K'
            scaled_volume = volume / 1000
        ax2.bar(x, scaled_volume, color=candle_colors)
        volume_title = 'Volume'
        if volume_scale:
            volume_title = 'Volume (%s)' % volume_scale
        ax2.set_title(volume_title)
        ax2.xaxis.grid(False)
        