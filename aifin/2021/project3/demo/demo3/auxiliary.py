def generate_bar(data):
    ## Data is a pandas dataframe
    import pandas as pd
    open_price = data['open'][0]
    close = data['close'][len(data) - 1]
    high = data['high'].max()
    low = data['low'].min()
    volume_ave = data['volume'].mean()
    OHLC = pd.DataFrame(data = [[open_price, high, low, close, volume_ave]], columns = ['open', 'high', 'low', 'close', 'volume_ave'])

    return OHLC
