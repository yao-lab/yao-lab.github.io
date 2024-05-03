import pandas as pd
import numpy as np
import math
import warnings

warnings.filterwarnings("ignore")

df1 = pd.read_csv('datashare.csv')
df = pd.read_csv('PredictorData2023.csv')
df_ret = pd.read_csv('RET.csv')

for ind, st in enumerate(df['Index']):
    st = st.replace(',', '')
    df.loc[ind, 'Index'] = st
df['Index'] = df['Index'].astype(float)

for ind, st in enumerate(df_ret['RET']):
    if isinstance(st, str) and st.isupper():
        df_ret.loc[ind, 'RET'] = np.nan
df_ret['RET'] = df_ret['RET'].astype(float)

df['yyyymm'] = df['yyyymm'].astype(int)
df1['DATE'] = df1['DATE'].astype(int)
df1['DATE'] = df1['DATE']//100
df1 = df1.rename(columns = {'DATE': 'yyyymm'})
df_ret['date'] = df_ret['date'].astype(int)
df_ret['date'] = df_ret['date']//100
df_ret = df_ret.rename(columns = {'date': 'yyyymm', 'PERMNO': 'permno'})

df = df.loc[(df['yyyymm'] >= 195703) & (df['yyyymm'] < 201701)]
df1 = df1.loc[(df1['yyyymm'] >= 195703) & (df1['yyyymm'] < 201701)]
df_ret = df_ret.loc[(df_ret['yyyymm'] >= 195703) & (df_ret['yyyymm'] < 201701)]
df['dp'] = df['D12'].apply(math.log) - df['Index'].apply(math.log)
df['ep'] = df['E12'].apply(math.log) - df['Index'].apply(math.log)
df['tms'] = df['lty'] - df['tbl']
df['dfy'] = df['BAA'] - df['AAA']
#Rename b/m by bm and Keep only 8 needed macropredictors
mac_pre = ['yyyymm', 'dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy','svar']
df = df.rename(columns = {'b/m':'bm'})[mac_pre]

data_0 = pd.merge(df1, df, how = 'left', on = 'yyyymm', suffixes=('', '_macro'))
data_0['permno'] = data_0['permno'].astype('int64')
df_ret['permno'] = df_ret['permno'].astype('int64')
data_0['yyyymm'] = data_0['yyyymm'].astype('int64')
df_ret['yyyymm'] = df_ret['yyyymm'].astype('int64')
data = pd.merge(data_0, df_ret, how = 'left', on = ['yyyymm', 'permno'])

del df, df1, df_ret, data_0

date_no = ['yyyymm', 'permno'] #Date and permno
macro = ['dp', 'ep_macro', 'bm_macro', 'ntis', 'tbl', 'tms', 'dfy', 'svar'] #Macroeconomic predictors
sic2 = ['sic2'] #Industrial dummies
ret = ['RET'] #Returns
lst = date_no + macro + sic2 + ret
stock_cha = [p for p in data.columns if p not in lst] #Stock-level predictors

# Fill the missing value
date_list = np.unique(data['yyyymm'])
for cha in (stock_cha + ret):
    for date in date_list:
        index_n = (data['yyyymm'] == date)
        median_n = np.nanmedian(data.loc[index_n, cha])
        # Fill missing values with median
        data.loc[index_n, cha] = data.loc[index_n, cha].replace(np.nan, median_n)

del date_list

data['excess_ret'] = data['RET'] - data['tbl'] #Calculate return in excess of risk-free rate
data = data.drop(columns = ret)

data = data.fillna(0)

#Save the data after cleaning
data.to_csv('data_cleaned.csv',  index = False)





