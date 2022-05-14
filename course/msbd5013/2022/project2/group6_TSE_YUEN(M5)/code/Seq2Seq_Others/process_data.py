import numpy as np
import pandas as pd
import category_encoders as ce
import pickle as pkl
pd.options.mode.chained_assignment = None
from utils.reduce_mem import reduce_mem


def process_data(input_data_dir='./data/', output_dir='./processed_data'):
    train_data = pd.read_csv(f'{input_data_dir}/sales_train_evaluation.csv')
    sell_prices = pd.read_csv(f'{input_data_dir}/sell_prices.csv')
    calendar = pd.read_csv(f'{input_data_dir}/calendar.csv')

    # ---- process calendar features ---- #
    print('* Processing calendar features')

    calendar.date = pd.to_datetime(calendar.date)
    calendar['relative_year'] = 2016 - calendar.year

    # convert month, day and weekday to cyclic encodings
    calendar['month_sin'] = np.sin(2 * np.pi * calendar.month/12.0)
    calendar['month_cos'] = np.cos(2 * np.pi * calendar.month/12.0)
    calendar['day_sin'] = np.sin(2 * np.pi * calendar.date.dt.day/calendar.date.dt.days_in_month)
    calendar['day_cos'] = np.cos(2 * np.pi * calendar.date.dt.day/calendar.date.dt.days_in_month)
    calendar['weekday_sin'] = np.sin(2 * np.pi * calendar.wday/7.0)
    calendar['weekday_cos'] = np.cos(2 * np.pi * calendar.wday/7.0)

    # use same encoded labels for both the event name columns
    cal_label = ['event_name_1', 'event_name_2']
    cal_label_encoded_cols = ['event_name_1_enc', 'event_name_2_enc']
    calendar[cal_label_encoded_cols] = calendar[cal_label]
    cal_label_encoder = ce.OrdinalEncoder(cols=cal_label_encoded_cols) # encoder to assign unique id for each unique value
    cal_label_encoder.fit(calendar)
    cal_label_encoder.mapping[1]['mapping'] = cal_label_encoder.mapping[0]['mapping']
    calendar = cal_label_encoder.transform(calendar)
    
    # save calendar
    # reduce_mem(calendar).to_csv('./processed_data/calendar.csv', index=False)
    
    # subtract one from label encoded as pytorch uses 0-indexing
    for col in cal_label_encoded_cols:
        calendar[col] = calendar[col] - 1

    calendar_df = calendar[['wm_yr_wk', 'd', 'snap_CA', 'snap_TX', 'snap_WI', 'relative_year',
                            'month_sin', 'month_cos', 'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos']
                           + cal_label_encoded_cols]

    # ---- Merge all dfs, keep calender_df features separate and just concat them for each batch ---- #
    train_data.id = train_data.id.str[:-11]
    sell_prices['id'] = sell_prices['item_id'] + '_' + sell_prices['store_id']

    # add empty columns for future data
    train_data = pd.concat([train_data, pd.DataFrame(columns=['d_'+str(i) for i in range(1942, 1970)])])

    # Encode categorical features using either one-hot or label encoding (for embeddings)
    print('* Encoding categorical features')
    label = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    label_encoded_cols = [str(i)+'_enc' for i in label]

    train_data[label_encoded_cols] = train_data[label]
    label_encoder = ce.OrdinalEncoder(cols=[str(i)+'_enc' for i in label])
    label_encoder.fit(train_data)
    train_data = label_encoder.transform(train_data)

    # subtract one from label encoded as pytorch uses 0-indexing
    for col in label_encoded_cols:
        train_data[col] = train_data[col] - 1
    train_data = reduce_mem(train_data)
    # Reshape, change dtypes and add previous day sales
    print('* Add previous day sales and merge sell prices')
    data_df = pd.melt(train_data, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                                           'item_id_enc', 'dept_id_enc', 'cat_id_enc', 'store_id_enc', 'state_id_enc'],
                      var_name='d', value_vars=['d_'+str(i) for i in range(1, 1970)], value_name='sales')

    # change dtypes to reduce memory usage
    data_df[['sales']] = data_df[['sales']].fillna(-2).astype(np.int16)  # fill future sales as -2
    calendar_df[['snap_CA', 'snap_TX', 'snap_WI', 'relative_year']] = calendar_df[
        ['snap_CA', 'snap_TX', 'snap_WI', 'relative_year']].astype(np.int8)
    calendar_df[cal_label_encoded_cols] = calendar_df[cal_label_encoded_cols].astype(np.int16)

    data_df[label_encoded_cols] = data_df[label_encoded_cols].astype(np.int16)

    # merge sell prices
    data_df = data_df.merge(right=calendar_df[['d', 'wm_yr_wk']], on=['d'], how='left')
    data_df = data_df.merge(right=sell_prices[['id', 'wm_yr_wk', 'sell_price']], on=['id', 'wm_yr_wk'], how='left')

    data_df.sell_price = data_df.sell_price.fillna(0.0)
    data_df['prev_day_sales'] = data_df.groupby(['id'])['sales'].shift(1)

    # remove data for d_1
    data_df.dropna(axis=0, inplace=True)
    calendar_df = reduce_mem(calendar_df[calendar_df.d != 'd_1'])

    # change dtypes
    # data_Df is a dataframe with all unique items with its sales
    data_df[['prev_day_sales']] = data_df[['prev_day_sales']].astype(np.int16)
    
    # remove category columns
    del data_df['wm_yr_wk']
    del data_df['item_id']
    del data_df['dept_id']
    del data_df['cat_id']
    del data_df['store_id']
    del data_df['state_id']
    
    num_samples = data_df.id.nunique()
    num_timesteps = data_df.d.nunique()
    data_df = data_df.set_index(['id', 'd'])
    
    data_df = (reduce_mem(data_df))
    
    ids = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    enc_dec_feats = ['sell_price'] + label_encoded_cols
    enc_only_feats = data_df.columns.difference(['sales', 'sell_price', 'prev_day_sales'] + enc_dec_feats)
    
    # the unique id defined by 5 different ids to indentify a unique item
    sales_data_ids = train_data[ids].values # shape (30490, 5)
    
    Y = data_df.sales.values.reshape(num_timesteps, num_samples).T # sales value of all 30490 items in 1968 days
    X_enc_only_feats = np.array(data_df[enc_only_feats]).reshape(num_timesteps, num_samples, -1) # shape: (1968, 30490, 0)
    X_enc_dec_feats = np.array(data_df[enc_dec_feats]).reshape(num_timesteps, num_samples, -1) # features to be encode
    X_prev_day_sales = data_df.prev_day_sales.values.reshape(num_timesteps, num_samples) # shape: (1968, 30490) the previous price of each item in each day
    calendar_index = calendar_df.d
    X_calendar = np.array(calendar_df.iloc[:, 2:]) # the day info of each item in each day # shape: (1968*14)
    X_calendar_cols = list(calendar_df.columns[2:])
    
    '''print(sales_data_ids.shape)
    print(calendar_index.shape)
    print(X_prev_day_sales.shape)
    print(X_enc_only_feats)
    print(X_enc_only_feats.shape)
    print(X_enc_dec_feats.shape)
    print(len(enc_dec_feats))
    print(len(enc_only_feats))
    print(X_calendar.shape)
    #print(X_calendar_cols.shape)
    print(Y.shape)'''
    
    
    print('* Save processed data')
    data_dict = {'sales_data_ids': sales_data_ids, 'calendar_index': calendar_index,
                 'X_prev_day_sales': X_prev_day_sales,
                 'X_enc_only_feats': X_enc_only_feats, 'X_enc_dec_feats' : X_enc_dec_feats,
                 'enc_dec_feat_names': enc_dec_feats, 'enc_only_feat_names': enc_only_feats,
                 'X_calendar': X_calendar, 'X_calendar_cols': X_calendar_cols,
                 'Y': Y,
                 'cal_label_encoder': cal_label_encoder, 'label_encoder': label_encoder}
    
    # pickle data
    with open(f'{output_dir}/data.pickle', 'wb') as f:
        pkl.dump(data_dict, f, protocol=pkl.HIGHEST_PROTOCOL)