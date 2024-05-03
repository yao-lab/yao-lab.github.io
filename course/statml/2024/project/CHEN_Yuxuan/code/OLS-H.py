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

# OLS
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

    ols_0 = SGDRegressor(loss='huber', alpha=1e-3, epsilon=0.05, learning_rate='optimal')
    ols = make_pipeline(StandardScaler(), ols_0)
    ols.fit(x_train, y_train)

    R2_test = R2_OOS(y_test, ols.predict(x_test).reshape(-1))
    R2_year[ind] = R2_test
    ind += 1

    del x_test, y_test, ols_0, ols

ind_max = np.argmax(R2_year)
R2_max = R2_year[ind_max]
pd.DataFrame(R2_year).to_csv('OLSH_R2.csv',  index = False)
