import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from Processing_Function import split, R2_OOS
from sklearn.ensemble import GradientBoostingRegressor as GBR
import math

n_year = len( range(198701, 201701, 100) )
R2_year = np.zeros(n_year)
lr_year = np.zeros(n_year)
ind = 0
for test_date in range(198701, 201701, 100):
    data = pd.read_csv('data_cleaned.csv')
    train_ind, val_ind, test_ind = split(test_date, data['yyyymm'])
    df_train = data[train_ind]
    df_val = data[val_ind]
    df_test = data[test_ind]

    del data

    y_train = df_train['excess_ret']
    x_train = df_train.drop(['yyyymm', 'permno', 'excess_ret'], axis = 1)
    y_val = df_val['excess_ret']
    x_val = df_val.drop(['yyyymm', 'permno', 'excess_ret'], axis = 1)
    y_test = df_test['excess_ret']
    x_test = df_test.drop(['yyyymm', 'permno', 'excess_ret'], axis = 1)

    del df_train, df_val, df_test

    R2_max = float("-Inf")

    learning_rate = [0.01, 0.1]

    for lr in learning_rate:
        GBR_H = make_pipeline(StandardScaler(), GBR(max_depth = 2, learning_rate = lr, loss = 'huber', n_estimators = 30))
        GBR_H.fit(x_train, y_train)
        val_pred = GBR_H.predict(x_val).reshape(-1)
        R2 = R2_OOS(y_val, val_pred) #annual R2 score
        if R2 > R2_max:
            R2_max = R2
            lr_max = lr

        del GBR_H

    x_all = pd.concat([x_train, x_val])
    y_all = pd.concat([y_train, y_val])

    del x_train, y_train, x_val, y_val

    model_best = make_pipeline(StandardScaler(), GBR(max_depth = 2, learning_rate = lr_max, loss = 'huber', n_estimators = 30))
    model_best.fit(x_all, y_all)

    R2_test = R2_OOS(y_test, model_best.predict(x_test).reshape(-1))
    R2_year[ind] = R2_test
    lr_year[ind] = lr_max
    ind += 1

    del x_all, y_all, x_test, y_test, model_best

pd.DataFrame(R2_year).to_csv('GBRH_R2.csv',  index = False)
ind_max = np.argmax(R2_year)
lr_max = lr_year[ind_max]

#Variable Importance
year_best = 198701 + 100 * ind_max
data = pd.read_csv('data_cleaned.csv')
stock_top = np.unique(data['permno'])[999]
idx_top = ( data['permno'] <= stock_top )
data = data[idx_top]
train_ind, val_ind, test_ind = split(year_best, data['yyyymm'])
df_train = pd.concat([ data[train_ind], data[val_ind] ])
df_test = data[test_ind]

del data

y_train = df_train['excess_ret']
x_train_0 = df_train.drop(['yyyymm', 'permno', 'excess_ret'], axis = 1)
y_test = df_test['excess_ret']
x_test_0 = df_test.drop(['yyyymm', 'permno', 'excess_ret'], axis = 1)

del df_train, df_test

model_best = make_pipeline(StandardScaler(), GBR(max_depth = 2, learning_rate = lr_max, loss = 'huber', n_estimators = 30))
model_best.fit(x_train_0, y_train)

R2_max = R2_OOS(y_test, model_best.predict(x_test_0).reshape(-1))

col_name = x_train_0.columns
macro = ['dp', 'ep_macro', 'bm_macro', 'ntis', 'tbl', 'tms', 'dfy', 'svar']
Im = []
sum_1 = 0
sum_2 = 0
for name in col_name:
    x_train = x_train_0.drop(columns = name)
    x_test = x_test_0.drop(columns = name)
    GBR_H = make_pipeline(StandardScaler(), GBR(max_depth = 2, learning_rate = lr_max, loss = 'huber', n_estimators = 30))
    GBR_H.fit(x_train, y_train)
    R2_name = R2_OOS(y_test, GBR_H.predict(x_test).reshape(-1))
    Im.append(abs(R2_name - R2_max))
    if name in macro:
        sum_2 += abs(R2_name - R2_max)
    else:
        sum_1 += abs(R2_name - R2_max)

    del x_train, x_test

for j, name in enumerate(col_name):
    if name in macro:
        Im[j] = Im[j] / sum_2
    else:
        Im[j] = Im[j] / sum_1

Im = pd.DataFrame( np.array(Im).reshape( (1,-1) ) )
Im.columns = col_name
Im.to_csv('GBRH_Importance.csv',  index = False)