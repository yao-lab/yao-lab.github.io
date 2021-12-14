import os
import pandas as pd
import numpy as np
import gc
import copy
import datetime
import matplotlib.pyplot as plt
from tqdm import trange


# Parameter
base_path = ('/').join(os.getcwd().split('\\'))


# Load Data
# calendar = pd.read_csv(f'{base_path}/dataset/calendar.csv')
# sales = pd.read_csv(f'{base_path}/dataset/sales_train_evaluation.csv')
# sales.to_csv(f'{base_path}/dataset/sales_train_evaluation_head100.csv')
# sales = pd.read_csv(f'{base_path}/dataset/sales_train_evaluation_head100.csv', index_col=0)
# prices = pd.read_csv(f'{base_path}/dataset/sell_prices.csv')
# prices.to_csv(f'{base_path}/dataset/sell_prices_head100.csv')
# prices = pd.read_csv(f'{base_path}/dataset/sell_prices_head100.csv', index_col=0)


# Process
# ########################################################################################################################
# # 总处理
# data1 = sales.drop(columns=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis=1).set_index('id')
# del sales; gc.collect()
# data1 = data1.unstack().reset_index()
# data1.columns = ['day', 'id', 'sales']
# data1['day'] = data1['day'].apply(lambda x: int(x.split('_')[-1]) + 28)
# data1['week'] = data1['day'].apply(lambda x: x//7 + 1)
# data1['time'] = data1['week'].apply(lambda x: pd.to_datetime(str(2011+x//52)+'-01-01') + datetime.timedelta(days = 7*(x%52)))
# data1 = data1.groupby(['id', 'time'])['sales'].sum().reset_index()
# data1.to_csv(f'{base_path}/dataset/总处理/data1.csv')
# # data1 = pd.read_csv(f'{base_path}/dataset/总处理/data1.csv', index_col=0)
#
# data2 = copy.deepcopy(prices)
# del prices; gc.collect()
# data2['id'] = data2['item_id'] + '_' +data2['store_id'] + '_evaluation'
# data2 = data2.drop(columns=['store_id', 'item_id'], axis=1)
# data2['time'] = data2['wm_yr_wk'].apply(lambda x: pd.to_datetime('20'+str(x)[1:3]+'-01-01') + datetime.timedelta(days = 7*int(str(x)[-2:])))
# data2 = data2.drop(columns=['wm_yr_wk'], axis=0)
# data2.to_csv(f'{base_path}/dataset/总处理/data2.csv')
# # data2 = pd.read_csv(f'{base_path}/dataset/总处理/data2.csv', index_col=0)
#
# data = pd.merge(data1, data2, on=['id', 'time'], how='left')
# del data1, data2; gc.collect()
# data['money'] = data['sales'] * data['sell_price']
# data['dept_id'] = data['id'].apply(lambda x: ('_').join(x.split('_')[:2]))
# data['item_id'] = data['id'].apply(lambda x: ('_').join(x.split('_')[:3]))
# data['cat_id'] = data['id'].apply(lambda x: ('_').join(x.split('_')[:1]))
# data['store_id'] = data['id'].apply(lambda x: ('_').join(x.split('_')[3:5]))
# data['state_id'] = data['id'].apply(lambda x: ('_').join(x.split('_')[3:4]))
# data.to_csv(f'{base_path}/dataset/总处理/data.csv')
# del data; gc.collect()
# ########################################################################################################################



# ########################################################################################################################
# # 总销售额
# data = pd.read_csv(f'{base_path}/dataset/总处理/data.csv', index_col=0)
# res = data.groupby('time')['money'].sum().reset_index()
# res.to_excel(f'{base_path}/dataset/总销售额.xlsx')
# del data, res; gc.collect()
# ########################################################################################################################


# ########################################################################################################################
# # 不同state总销售额
# data = pd.read_csv(f'{base_path}/dataset/总处理/data.csv', index_col=0)
# data = data.groupby(['time', 'state_id'])['money'].sum().reset_index()
#
# res1 = data[data['state_id'] == 'CA']
# res1['time'] = res1['time'].apply(lambda x: str(x)[:7])
# res1 = res1.groupby('time')['money'].sum()
# res1 = res1.sort_index()
# res1.to_excel(f'{base_path}/dataset/不同state总销售额_CA.xlsx')
# del res1; gc.collect()
#
# res2 = data[data['state_id'] == 'TX']
# res2['time'] = res2['time'].apply(lambda x: str(x)[:7])
# res2 = res2.groupby('time')['money'].sum()
# res2 = res2.sort_index()
# res2.to_excel(f'{base_path}/dataset/不同state总销售额_TX.xlsx')
# del res2; gc.collect()
#
# res3 = data[data['state_id'] == 'WI']
# res3['time'] = res3['time'].apply(lambda x: str(x)[:7])
# res3 = res3.groupby('time')['money'].sum()
# res3 = res3.sort_index()
# res3.to_excel(f'{base_path}/dataset/不同state总销售额_WI.xlsx')
# del res3; gc.collect()
# del data; gc.collect()
# ########################################################################################################################



# ########################################################################################################################
# # 不同cat总销售额
# data = pd.read_csv(f'{base_path}/dataset/总处理/data.csv', index_col=0)
# data = data.groupby(['time', 'cat_id'])['money'].sum().reset_index()
#
# df = pd.DataFrame()
#
# res1 = data[data['cat_id'] == 'HOBBIES']
# res1['time'] = res1['time'].apply(lambda x: str(x)[:7])
# res1 = res1.groupby('time')['money'].sum()
# res1 = res1.sort_index()
# res1 = res1.reset_index()
# res1.to_excel(f'{base_path}/dataset/不同cat总销售额_HOBBIES.xlsx')
# df['HOBBIES'] = [res1['money'].sum()]
# del res1; gc.collect()
#
# res2 = data[data['cat_id'] == 'FOODS']
# res2['time'] = res2['time'].apply(lambda x: str(x)[:7])
# res2 = res2.groupby('time')['money'].sum()
# res2 = res2.sort_index()
# res2 = res2.reset_index()
# res2.to_excel(f'{base_path}/dataset/不同cat总销售额_FOODS.xlsx')
# df['FOODS'] = [res2['money'].sum()]
# del res2; gc.collect()
#
# res3 = data[data['cat_id'] == 'HOUSEHOLD']
# res3['time'] = res3['time'].apply(lambda x: str(x)[:7])
# res3 = res3.groupby('time')['money'].sum()
# res3 = res3.sort_index()
# res3 = res3.reset_index()
# res3.to_excel(f'{base_path}/dataset/不同cat总销售额_HOUSEHOLD.xlsx')
# df['HOUSEHOLD'] = [res3['money'].sum()]
# del res3; gc.collect()
#
# df.to_excel(f'{base_path}/dataset/不同cat_MoneyPerCat.xlsx')
# del data, df; gc.collect()
# ########################################################################################################################



#
# ########################################################################################################################
# # 不同store总销售额
# data = pd.read_csv(f'{base_path}/dataset/总处理/data.csv', index_col=0)
# data = data.groupby(['time', 'store_id'])['money'].sum().reset_index().set_index('time')
#
# CA_1 = data[data['store_id'] == 'CA_1'][['money']]
# CA_1.columns = ['CA_1']
# CA_2 = data[data['store_id'] == 'CA_2'][['money']]
# CA_2.columns = ['CA_2']
# CA_3 = data[data['store_id'] == 'CA_3'][['money']]
# CA_3.columns = ['CA_3']
# CA_4 = data[data['store_id'] == 'CA_4'][['money']]
# CA_4.columns = ['CA_4']
# CA = pd.concat([CA_1, CA_2, CA_3, CA_4], axis=1, sort=True)
# CA['time1'] = CA.index
# CA['time1'] = CA['time1'].apply(lambda x: str(x)[:7])
# CA = CA.groupby('time1')['CA_1', 'CA_2', 'CA_3', 'CA_4'].sum()
# CA = CA.sort_index()
# CA.to_excel(f'{base_path}/dataset/不同store总销售额_CA.xlsx')
# del CA_1, CA_2, CA_3, CA_4, CA; gc.collect()
#
# TX_1 = data[data['store_id'] == 'TX_1'][['money']]
# TX_1.columns = ['TX_1']
# TX_2 = data[data['store_id'] == 'TX_2'][['money']]
# TX_2.columns = ['TX_2']
# TX_3 = data[data['store_id'] == 'TX_3'][['money']]
# TX_3.columns = ['TX_3']
# TX = pd.concat([TX_1, TX_2, TX_3], axis=1, sort=True)
# TX['time1'] = TX.index
# TX['time1'] = TX['time1'].apply(lambda x: str(x)[:7])
# TX = TX.groupby('time1')['TX_1', 'TX_2', 'TX_3'].sum()
# TX = TX.sort_index()
# TX.to_excel(f'{base_path}/dataset/不同store总销售额_TX.xlsx')
# del TX_1, TX_2, TX_3, TX; gc.collect()
#
# WI_1 = data[data['store_id'] == 'WI_1'][['money']]
# WI_1.columns = ['WI_1']
# WI_2 = data[data['store_id'] == 'WI_2'][['money']]
# WI_2.columns = ['WI_2']
# WI_3 = data[data['store_id'] == 'WI_3'][['money']]
# WI_3.columns = ['WI_3']
# WI = pd.concat([WI_1, WI_2, WI_3], axis=1, sort=True)
# WI['time1'] = WI.index
# WI['time1'] = WI['time1'].apply(lambda x: str(x)[:7])
# WI = WI.groupby('time1')['WI_1', 'WI_2', 'WI_3'].sum()
# WI = WI.sort_index()
# WI.to_excel(f'{base_path}/dataset/不同store总销售额_WI.xlsx')
# del WI_1, WI_2, WI_3, WI; gc.collect()
# del data; gc.collect()
# ########################################################################################################################


# ########################################################################################################################
# # 对总销量进行季节调整
# data = pd.read_excel(f'{base_path}/dataset/数据结果/总销售额.xlsx', index_col=0)
# data['time'] = data['time'].apply(lambda x: str(x)[:7])
# data = data.groupby('time')['sales'].sum().reset_index().set_index('time')
# df = pd.DataFrame()
#
# import statsmodels.api as sm
# from statsmodels.tsa.seasonal import seasonal_decompose
# import statsmodels.tsa.stattools as ts
# decomposition = seasonal_decompose(data, model='multiplicative', two_sided=False, freq=12)
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid
# df['trend'] = trend
# df['seasonal'] = seasonal
# df['residual'] = residual
# # plt.figure(figsize=[15, 7])
# # decomposition.plot()
# # 提取趋势项以及循环项
# data_TC = trend
# # HP滤波
# cycle, trend = sm.tsa.filters.hpfilter(data_TC.dropna(), lamb=14400)
# cycle = data_TC / trend
# df['cycle'] = cycle
# # 转换为同比口径
# YOY = cycle / cycle.shift(12) - 1
# df['YOY'] = YOY
# # YOY.plot()
# df.to_excel(f'{base_path}/dataset/总销售额季节调整.xlsx')
# del YOY, cycle, data_TC, data, decomposition, df, residual, seasonal, trend; gc.collect()
# ########################################################################################################################



# ########################################################################################################################
# # 生成总处理 data_daily 数据
# # calendar = pd.read_csv(f'{base_path}/dataset/calendar.csv')
# # data = pd.read_csv(f'{base_path}/dataset/sales_train_evaluation.csv')
# # # data = pd.read_csv(f'{base_path}/dataset/sales_train_evaluation_head100.csv', index_col=0)
# # data = data.drop(columns=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis=1).set_index('id')
# # data = data.unstack().reset_index()
# # data.columns = ['date', 'id', 'sales']
# # data['date'] = data['date'].apply(lambda x: str(pd.to_datetime('2011-01-28')+datetime.timedelta(int(x.split('_')[-1])))[:10])
# # data.to_csv(f'{base_path}/dataset/总处理/data_daily.csv')
# # del data; gc.collect()
# #
# # data = pd.read_csv(f'{base_path}/dataset/总处理/data_daily.csv', index_col=0)
# # total = data.shape[0]
# # # total = 156283
# # num = 100
# num = 25
# # from tqdm import trange
# # for i in trange(1, num+1):
# #     # print(int((i - 1) * total // num), int((i) * total // num))
# #     tmp = pd.merge(data[int((i - 1) * total // num): int((i) * total // num)], calendar, how='left', on=['date'])
# #     tmp.to_csv(f'{base_path}/dataset/总处理/data_daily{i}.csv')
# # del calendar, data, tmp, total; gc.collect()
#
# data = []
# # # for i in trange(1, num+1):
# for i in trange(76, 100+1):
#     tmp = pd.read_csv(f'{base_path}/dataset/总处理/data_daily{i}.csv', index_col=0, low_memory=False)
#     data.append(tmp)
# data = pd.concat(data, axis=0)
# data.to_csv(f'{base_path}/dataset/总处理/data_daily.csv')
# del data, tmp; gc.collect()
#
# data = pd.read_csv(f'{base_path}/dataset/总处理/data_daily.csv', index_col=0)
# data['dept_id'] = data['id'].apply(lambda x: ('_').join(x.split('_')[:2]))
# data['item_id'] = data['id'].apply(lambda x: ('_').join(x.split('_')[:3]))
# data['cat_id'] = data['id'].apply(lambda x: ('_').join(x.split('_')[:1]))
# data['store_id'] = data['id'].apply(lambda x: ('_').join(x.split('_')[3:5]))
# data['state_id'] = data['id'].apply(lambda x: ('_').join(x.split('_')[3:4]))
# data.to_csv(f'{base_path}/dataset/总处理/data_daily.csv')
# del data; gc.collect()
# ########################################################################################################################



# ########################################################################################################################
# # week 和 month “销量” 探究
# data = pd.read_csv(f'{base_path}/dataset/总处理/data_daily.csv', index_col=0, low_memory=False)
# data = data.groupby(['wday', 'month'])['sales'].sum().reset_index()
# data.to_excel(f'{base_path}/dataset/week&month销量探究.xlsx')
# del data; gc.collect()
# ########################################################################################################################


# ########################################################################################################################
# # 是否有event对 “销量” 影响
# data = pd.read_csv(f'{base_path}/dataset/总处理/data_daily.csv', index_col=0)[['date', 'sales', 'cat_id', 'event_name_1',
#                                                                             'event_type_1', 'event_name_2', 'event_type_2']]
#
# res1 = data[data['cat_id'] == 'HOBBIES']
# res1['event_type_1'] = res1['event_type_1'].fillna(0.0)
# res1_T = res1[res1['event_type_1'] != 0.0]
# res1_T = res1_T.groupby('date')['sales'].sum().reset_index().set_index('date')
# res1_T.columns = ['HOBBIES_Event']
# res1_F = res1[res1['event_type_1'] == 0.0]
# res1_F = res1_F.groupby('date')['sales'].sum().reset_index().set_index('date')
# res1_F.columns = ['HOBBIES_NoEvent']
# res1 = pd.concat([res1_T, res1_F], axis=1, sort=True)
# res1.to_excel(f'{base_path}/dataset/HOBBIES_Event销量探究.xlsx')
# del res1, res1_F, res1_T; gc.collect()
#
# res2 = data[data['cat_id'] == 'FOODS']
# res2['event_type_1'] = res2['event_type_1'].fillna(0.0)
# res2_T = res2[res2['event_type_1'] != 0.0]
# res2_T = res2_T.groupby('date')['sales'].sum().reset_index().set_index('date')
# res2_T.columns = ['FOODS_Event']
# res2_F = res2[res2['event_type_1'] == 0.0]
# res2_F = res2_F.groupby('date')['sales'].sum().reset_index().set_index('date')
# res2_F.columns = ['FOODS_NoEvent']
# res2 = pd.concat([res2_T, res2_F], axis=1, sort=True)
# res2.to_excel(f'{base_path}/dataset/FOODS_Event销量探究.xlsx')
# del res2, res2_F, res2_T; gc.collect()
#
# res3 = data[data['cat_id'] == 'HOUSEHOLD']
# res3['event_type_1'] = res3['event_type_1'].fillna(0.0)
# res3_T = res3[res3['event_type_1'] != 0.0]
# res3_T = res3_T.groupby('date')['sales'].sum().reset_index().set_index('date')
# res3_T.columns = ['HOUSEHOLD_Event']
# res3_F = res3[res3['event_type_1'] == 0.0]
# res3_F = res3_F.groupby('date')['sales'].sum().reset_index().set_index('date')
# res3_F.columns = ['HOUSEHOLD_NoEvent']
# res3 = pd.concat([res3_T, res3_F], axis=1, sort=True)
# res3.to_excel(f'{base_path}/dataset/HOUSEHOLD_Event销量探究.xlsx')
# del res3, res3_F, res3_T; gc.collect()
# del data; gc.collect()
# ########################################################################################################################



# ########################################################################################################################
# # 把 FOODS_Event销量探究、HOBBIES_Event销量探究、HOUSEHOLD_Event销量探究、week&month销量探究 1234 组合起来
# group = 4
# res1 = []
# res2 = []
# res3 = []
# res4 = []
# for i in range(1, group+1):
#     tmp1 = pd.read_excel(f'{base_path}/dataset/daily/FOODS_Event销量探究{i}.xlsx', index_col=0)
#     tmp2 = pd.read_excel(f'{base_path}/dataset/daily/HOBBIES_Event销量探究{i}.xlsx', index_col=0)
#     tmp3 = pd.read_excel(f'{base_path}/dataset/daily/HOUSEHOLD_Event销量探究{i}.xlsx', index_col=0)
#     tmp4 = pd.read_excel(f'{base_path}/dataset/daily/week&month销量探究{i}.xlsx', index_col=0)
#     tmp1.columns = [f'{col}_{i}' for col in tmp1.columns]
#     tmp2.columns = [f'{col}_{i}' for col in tmp2.columns]
#     tmp3.columns = [f'{col}_{i}' for col in tmp3.columns]
#     tmp4.columns = [f'{col}_{i}' for col in tmp4.columns]
#     res1.append(tmp1)
#     res2.append(tmp2)
#     res3.append(tmp3)
#     res4.append(tmp4)
# res1 = pd.concat(res1, axis=1, sort=True).fillna(0.0)
# res2 = pd.concat(res2, axis=1, sort=True).fillna(0.0)
# res3 = pd.concat(res3, axis=1, sort=True).fillna(0.0)
# res4 = pd.concat(res4, axis=1, sort=True).fillna(0.0)
#
# res1['FOODS_Event'] = res1['FOODS_Event_1'] + res1['FOODS_Event_2'] + res1['FOODS_Event_3'] + res1['FOODS_Event_4']
# res1['FOODS_NoEvent'] = res1['FOODS_NoEvent_1'] + res1['FOODS_NoEvent_2'] + res1['FOODS_NoEvent_3'] + res1['FOODS_NoEvent_4']
# res1 = res1[['FOODS_Event', 'FOODS_NoEvent']]
# res2['HOBBIES_Event'] = res2['HOBBIES_Event_1'] + res2['HOBBIES_Event_2'] + res2['HOBBIES_Event_3'] + res2['HOBBIES_Event_4']
# res2['HOBBIES_NoEvent'] = res2['HOBBIES_NoEvent_1'] + res2['HOBBIES_NoEvent_2'] + res2['HOBBIES_NoEvent_3'] + res2['HOBBIES_NoEvent_4']
# res2 = res2[['HOBBIES_Event', 'HOBBIES_NoEvent']]
# res3['HOUSEHOLD_Event'] = res3['HOUSEHOLD_Event_1'] + res3['HOUSEHOLD_Event_2'] + res3['HOUSEHOLD_Event_3'] + res3['HOUSEHOLD_Event_4']
# res3['HOUSEHOLD_NoEvent'] = res3['HOUSEHOLD_NoEvent_1'] + res3['HOUSEHOLD_NoEvent_2'] + res3['HOUSEHOLD_NoEvent_3'] + res3['HOUSEHOLD_NoEvent_4']
# res3 = res3[['HOUSEHOLD_Event', 'HOUSEHOLD_NoEvent']]
# res4['sales'] = res4['sales_1'] + res4['sales_2'] + res4['sales_3'] + res4['sales_4']
# res4['wday'] = res4['wday_1']
# res4['month'] = res4['month_1']
# res4 = res4[['wday', 'month', 'sales']]
#
# res1.index = pd.to_datetime(res1.index)
# res1.replace(0.0, np.nan).to_excel(f'{base_path}/dataset/FOODS_Event销量探究.xlsx')
# res2.index = pd.to_datetime(res2.index)
# res2.replace(0.0, np.nan).to_excel(f'{base_path}/dataset/HOBBIES_Event销量探究.xlsx')
# res3.index = pd.to_datetime(res3.index)
# res3.replace(0.0, np.nan).to_excel(f'{base_path}/dataset/HOUSEHOLD_Event销量探究.xlsx')
# res4 = res4.pivot_table(index=['wday'], columns=['month'], values=['sales'])
# res4.to_excel(f'{base_path}/dataset/week&month销量探究.xlsx')
# del res1, res2, res3, tmp1, tmp2, tmp3; gc.collect()
# ########################################################################################################################


# ########################################################################################################################
# # 探究 snap 对销量的影响
# # i = 1
# for i in trange(1, 5):
#     data = pd.read_csv(f'{base_path}/dataset/daily/data_daily{i}.csv', index_col=0, low_memory=False)
#     data['state_id'] = data['id'].apply(lambda x: ('_').join(x.split('_')[3:4]))
#
#
#     res1 = data[data['state_id'] == 'TX']
#     res1_Y = res1[res1['snap_TX'] != 0]
#     res1_N = res1[res1['snap_TX'] == 0]
#     res1_Y = res1_Y.groupby('date')['sales'].sum().reset_index().set_index('date')
#     res1_Y.columns = ['SNAP']
#     res1_N = res1_N.groupby('date')['sales'].sum().reset_index().set_index('date')
#     res1_N.columns = ['No_SNAP']
#     res1 = pd.concat([res1_Y, res1_N], axis=1, sort=True)
#     res1.to_excel(f'{base_path}/dataset/snap_TX_{i}.xlsx')
#
#     res2 = data[data['state_id'] == 'CA']
#     res2_Y = res2[res2['snap_CA'] != 0]
#     res2_N = res2[res2['snap_CA'] == 0]
#     res2_Y = res2_Y.groupby('date')['sales'].sum().reset_index().set_index('date')
#     res2_Y.columns = ['SNAP']
#     res2_N = res2_N.groupby('date')['sales'].sum().reset_index().set_index('date')
#     res2_N.columns = ['No_SNAP']
#     res2 = pd.concat([res2_Y, res2_N], axis=1, sort=True)
#     res2.to_excel(f'{base_path}/dataset/snap_CA_{i}.xlsx')
#
#     res3 = data[data['state_id'] == 'WI']
#     res3_Y = res3[res3['snap_WI'] != 0]
#     res3_N = res3[res3['snap_WI'] == 0]
#     res3_Y = res3_Y.groupby('date')['sales'].sum().reset_index().set_index('date')
#     res3_Y.columns = ['SNAP']
#     res3_N = res3_N.groupby('date')['sales'].sum().reset_index().set_index('date')
#     res3_N.columns = ['No_SNAP']
#     res3 = pd.concat([res3_Y, res3_N], axis=1, sort=True)
#     res3.to_excel(f'{base_path}/dataset/snap_WI_{i}.xlsx')
# del res1, res1_N, res1_Y, res1, res1_N, res1_Y, res3, res3_N, res3_Y
# ########################################################################################################################


########################################################################################################################
# 合并 snap 对不同州的影响
group = 4
res1 = []
res2 = []
res3 = []
for i in range(1, group+1):
    tmp1 = pd.read_excel(f'{base_path}/dataset/snap/snap_TX_{i}.xlsx', index_col=0)
    tmp2 = pd.read_excel(f'{base_path}/dataset/snap/snap_CA_{i}.xlsx', index_col=0)
    tmp3 = pd.read_excel(f'{base_path}/dataset/snap/snap_WI_{i}.xlsx', index_col=0)
    tmp1.columns = [f'{col}_{i}' for col in tmp1.columns]
    tmp2.columns = [f'{col}_{i}' for col in tmp2.columns]
    tmp3.columns = [f'{col}_{i}' for col in tmp3.columns]
    res1.append(tmp1)
    res2.append(tmp2)
    res3.append(tmp3)
res1 = pd.concat(res1, axis=1, sort=True).fillna(0.0)
res2 = pd.concat(res2, axis=1, sort=True).fillna(0.0)
res3 = pd.concat(res3, axis=1, sort=True).fillna(0.0)

res1['SNAP'] = res1['SNAP_1'] + res1['SNAP_2'] + res1['SNAP_3'] + res1['SNAP_4']
res1['No_SNAP'] = res1['No_SNAP_1'] + res1['No_SNAP_2'] + res1['No_SNAP_3'] + res1['No_SNAP_4']
res1 = res1[['SNAP', 'No_SNAP']]
res2['SNAP'] = res2['SNAP_1'] + res2['SNAP_2'] + res2['SNAP_3'] + res2['SNAP_4']
res2['No_SNAP'] = res2['No_SNAP_1'] + res2['No_SNAP_2'] + res2['No_SNAP_3'] + res2['No_SNAP_4']
res2 = res2[['SNAP', 'No_SNAP']]
res3['SNAP'] = res3['SNAP_1'] + res3['SNAP_2'] + res3['SNAP_3'] + res3['SNAP_4']
res3['No_SNAP'] = res3['No_SNAP_1'] + res3['No_SNAP_2'] + res3['No_SNAP_3'] + res3['No_SNAP_4']
res3 = res3[['SNAP', 'No_SNAP']]

res1.index = pd.to_datetime(res1.index)
res1.replace(0.0, np.nan).to_excel(f'{base_path}/dataset/snap_TX.xlsx')
res2.index = pd.to_datetime(res2.index)
res2.replace(0.0, np.nan).to_excel(f'{base_path}/dataset/snap_CA.xlsx')
res3.index = pd.to_datetime(res3.index)
res3.replace(0.0, np.nan).to_excel(f'{base_path}/dataset/snap_WI.xlsx')
# del res1, res2, res3, tmp1, tmp2, tmp3; gc.collect()


res3 = res3.replace(0.0, np.nan)
res3.mean(axis=0)



