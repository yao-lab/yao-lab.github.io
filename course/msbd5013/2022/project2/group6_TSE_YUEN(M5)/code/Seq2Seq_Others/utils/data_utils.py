import numpy as np
import pandas as pd


def get_aggregated_series(sales, sales_data_ids, agg_fn='sum'):
    """
    Aggregates 30,490 level 12 series to generate data for all 42,840 series

    Input data format:
    sales: np array of shape (30490, num_timesteps)
    sales_data_ids: np array of shape (30490, 5)
                    with 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id' as the columns
    agg_fn: function to be used for getting aggregated series' values ('mean' or 'sum')
    """

    df = pd.DataFrame({col: sales_data_ids[:, i] for col, i in
                       zip(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], range(0, 5))})
    df = pd.concat([df, pd.DataFrame(sales)], axis=1)
    data_cols = [i for i in range(0, sales.shape[1])]

    agg_indices, agg_series, agg_series_id = [], [], []

    # Level 1
    agg = np.sum(sales, 0) if agg_fn == 'sum' else np.mean(sales, 0)
    agg_series.append(agg.reshape(1, -1))
    agg_series_id.append(np.array(['Level1_Total_X']))

    # Level 2
    agg = df.groupby(['state_id'])[data_cols]
    agg_indices.append(agg.indices)
    agg = agg.agg(agg_fn)
    agg_series.append(agg.values)
    agg_series_id.append(('Level2_' + agg.index.values + '_X'))

    # Level 3
    agg = df.groupby(['store_id'])[data_cols]
    agg_indices.append(agg.indices)
    agg = agg.agg(agg_fn)
    agg_series.append(agg.values)
    agg_series_id.append(('Level3_' + agg.index.values + '_X'))

    # Level 4
    agg = df.groupby(['cat_id'])[data_cols]
    agg_indices.append(agg.indices)
    agg = agg.agg(agg_fn)
    agg_series.append(agg.values)
    agg_series_id.append(('Level4_' + agg.index.values + '_X'))

    # Level 5
    agg = df.groupby(['dept_id'])[data_cols]
    agg_indices.append(agg.indices)
    agg = agg.agg(agg_fn)
    agg_series.append(agg.values)
    agg_series_id.append(('Level5_' + agg.index.values + '_X'))

    # Level 6
    agg = df.groupby(['state_id', 'cat_id'])[data_cols]
    agg_indices.append(agg.indices)
    agg = agg.agg(agg_fn)
    agg_series.append(agg.values)
    agg_series_id.append('Level6_' + agg.index.get_level_values(0) + '_' + agg.index.get_level_values(1))

    # Level 7
    agg = df.groupby(['state_id', 'dept_id'])[data_cols]
    agg_indices.append(agg.indices)
    agg = agg.agg(agg_fn)
    agg_series.append(agg.values)
    agg_series_id.append('Level7_' + agg.index.get_level_values(0) + '_' + agg.index.get_level_values(1))

    # Level 8
    agg = df.groupby(['store_id', 'cat_id'])[data_cols]
    agg_indices.append(agg.indices)
    agg = agg.agg(agg_fn)
    agg_series.append(agg.values)
    agg_series_id.append('Level8_' + agg.index.get_level_values(0) + '_' + agg.index.get_level_values(1))

    # Level 9
    agg = df.groupby(['store_id', 'dept_id'])[data_cols]
    agg_indices.append(agg.indices)
    agg = agg.agg(agg_fn)
    agg_series.append(agg.values)
    agg_series_id.append('Level9_' + agg.index.get_level_values(0) + '_' + agg.index.get_level_values(1))

    # Level 10
    agg = df.groupby(['item_id'])[data_cols]
    agg_indices.append(agg.indices)
    agg = agg.agg(agg_fn)
    agg_series.append(agg.values)
    agg_series_id.append(('Level10_' + agg.index.values + '_X'))

    # Level 11
    agg = df.groupby(['state_id', 'item_id'])[data_cols]
    agg_indices.append(agg.indices)
    agg = agg.agg(agg_fn)
    agg_series.append(agg.values)
    agg_series_id.append('Level11_' + agg.index.get_level_values(0) + '_' + agg.index.get_level_values(1))

    # Level 12
    agg = df.set_index(['item_id', 'store_id'])[data_cols]
    agg_series.append(agg.values)
    agg_series_id.append('Level12_' + agg.index.get_level_values(0) + '_' + agg.index.get_level_values(1))

    # Get affected_hierarchy_ids - all the series affected on updating each Level 12 series
    affected_hierarchy_ids = np.empty((30490, 12), np.int32)

    # Level 1
    affected_hierarchy_ids[:, 0] = 0
    fill_id, fill_col = 1, 1
    # Level 2
    for k, v in agg_indices[0].items():
        affected_hierarchy_ids[v, fill_col] = fill_id
        fill_id += 1
    fill_col += 1
    # Level 3
    for k, v in agg_indices[1].items():
        affected_hierarchy_ids[v, fill_col] = fill_id
        fill_id += 1
    fill_col += 1
    # Level 4
    for k, v in agg_indices[2].items():
        affected_hierarchy_ids[v, fill_col] = fill_id
        fill_id += 1
    fill_col += 1
    # Level 5
    for k, v in agg_indices[3].items():
        affected_hierarchy_ids[v, fill_col] = fill_id
        fill_id += 1
    fill_col += 1
    # Level 6
    for k, v in agg_indices[4].items():
        affected_hierarchy_ids[v, fill_col] = fill_id
        fill_id += 1
    fill_col += 1
    # Level 7
    for k, v in agg_indices[5].items():
        affected_hierarchy_ids[v, fill_col] = fill_id
        fill_id += 1
    fill_col += 1
    # Level 8
    for k, v in agg_indices[6].items():
        affected_hierarchy_ids[v, fill_col] = fill_id
        fill_id += 1
    fill_col += 1
    # Level 9
    for k, v in agg_indices[7].items():
        affected_hierarchy_ids[v, fill_col] = fill_id
        fill_id += 1
    fill_col += 1
    # Level 10
    for k, v in agg_indices[8].items():
        affected_hierarchy_ids[v, fill_col] = fill_id
        fill_id += 1
    fill_col += 1
    # Level 11
    for k, v in agg_indices[9].items():
        affected_hierarchy_ids[v, fill_col] = fill_id
        fill_id += 1
    fill_col += 1
    # Level 12
    affected_hierarchy_ids[:, fill_col] = fill_id + np.arange(0, 30490)

    return np.concatenate(agg_series, axis=0), np.concatenate(agg_series_id, axis=0).\
        astype('<U28'), affected_hierarchy_ids


def get_weights_all_levels(sales, sell_price, sales_data_ids):
    """
    Generates weights for all 42,840 series

    Input data format:
    sales: np array of shape (30490, 28)
    sell_price: np array of shape (30490, 28)

    sales_data_ids: np array of shape (30490, 5)
                with 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id' as the columns
    """

    assert (sales.shape == sell_price.shape), "Sell price and Sales arrays have different sizes"
    assert (sales.shape[1] == 28), "Number of timesteps provided weight calculation is not equal to 28"

    # Get actual dollar sales for last 28 days for all 42,840 series
    dollar_sales = sales * sell_price
    agg_series, agg_series_id, _ = get_aggregated_series(dollar_sales, sales_data_ids)

    # Sum up the actual dollar sales for all 28 timesteps
    agg_series = agg_series.sum(1)

    # Calculate total sales for each level
    level_totals = agg_series[np.core.defchararray.find(agg_series_id, f'Level1_') == 0].sum()

    # Calculate weight for each series
    weights = agg_series / level_totals

    return weights, agg_series_id


def get_weights_level_12(sales, sell_price):
    """
    Generates weights for only 30,490 level 12 series

    Input data format:
    sales: np array of shape (30490, 28)
    sell_price: np array of shape (30490, 28)
    """

    assert (sales.shape == sell_price.shape), "Sell price and Sales arrays have different sizes"
    assert (sales.shape[1] == 28), "Number of timesteps provided weight calculation is not equal to 28"

    # Get actual dollar sales for last 28 days and sum them up
    dollar_sales = (sales * sell_price).sum(1)
    # Calculate weight for each series
    weights = dollar_sales / dollar_sales.sum()

    return weights


def update_preds_acc_hierarchy(prev_preds, preds, affected_ids):
    """
    prev_preds: Previously stored predictions for all 42,840 series (42840, n_timesteps)
    preds: Current batch predictions (batch_size, n_timesteps)
    affected_ids: the ids of all the series affected by the series in preds (30490, 12)
    """

    # get the change in predictions for the batch series
    change_preds = (preds - prev_preds[affected_ids[:, -1]]).repeat_interleave(12, dim=0)

    affected_ids = affected_ids.flatten()
    prev_preds = prev_preds.index_add(0, affected_ids, change_preds)

    return prev_preds
