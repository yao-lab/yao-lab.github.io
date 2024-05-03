import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from Processing_Function import split, R2_OOS
import math

n_year = len( range(198701, 201701, 100) )
R2_year = np.zeros(n_year)
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
    for lamb in [1e-4, 1e-3, 1e-2, 1e-1]:
        RIDGE_H = make_pipeline(StandardScaler(), SGDRegressor(loss = 'huber', penalty = 'l2', alpha = lamb, epsilon = 0.05, max_iter = 1e6, shuffle = False))
        RIDGE_H.fit(x_train, y_train)
        val_pred = RIDGE_H.predict(x_val).reshape(-1)
        R2 = R2_OOS(y_val, val_pred) #annual R2 score
        if R2 > R2_max:
            R2_max = R2
            lamb_max = lamb

        del RIDGE_H

    x_all = pd.concat([x_train, x_val])
    y_all = pd.concat([y_train, y_val])

    del x_train, y_train, x_val, y_val

    RI = SGDRegressor(loss = 'huber', penalty = 'l2', alpha = lamb_max, epsilon = 0.05, max_iter = 1e6, shuffle = False)
    model_best = make_pipeline(StandardScaler(), RI)
    model_best.fit(x_all, y_all)

    R2_test = R2_OOS(y_test, model_best.predict(x_test).reshape(-1))
    R2_year[ind] = R2_test
    ind += 1

    del x_test, y_test, model_best, RI

pd.DataFrame(R2_year).to_csv('RIDGEH_R2.csv',  index = False)