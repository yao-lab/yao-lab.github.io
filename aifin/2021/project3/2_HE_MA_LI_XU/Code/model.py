# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:29:41 2021

@author: majingkun
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
def data_clean(data, v_type):
    x = []
    grouped = data.groupby(data['id'])
    if v_type == 'x' :
        for name, group in grouped:
            print(name)
            x.append(np.array(group.drop(['id', 'd'], axis = 1)).T)
    else:
        for name, group in grouped:
            print(name)
            x.append(np.array(group.drop(['id'], axis = 1)))
    return x

def mse(y_test, y_pre):
    return np.mean((y_test - y_pre)**2)

def r_oos(y_test, y_pre):
    sst = np.sum((y_test - np.mean(y_test))**2)
    sse = np.sum((y_test - y_pre)**2)
    return 1 - sse / sst

if __name__ == '__main__':
    ################### clean x ############################
    d_raw = pd.read_csv(r'C:\Users\Administrator\Desktop\m5\data_final_s.csv')
    id_ = np.unique(d_raw['id'])
    id_1 = []
    x = data_clean(d_raw, 'x')
    
    ################## clean y ##############################
    y_raw = pd.read_csv(r'C:\Users\Administrator\Desktop\m5\sales_train_evaluation.csv')
    y1 = y_raw.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis = 1)
    y = data_clean(y1, 'y')
    
    
    ################## model ###########################
    ## regression
    import statsmodels.api as sm
    # x_1 = x[0][:,:913]
    # y_1 = y[0][0, 1000: 1913]
    # m = sm.OLS(y_1, sm.add_constant(x_1.T)).fit()
    # y_test = y[0][0, 1913:]
    # y_pre = m.predict(x[0][:, 913: 941].T)
    # print(m.summary())
    
    # r_all = []
    # y_pre_all = []
    y_v = []
    y_e = []
    for j in tqdm(range(len(x))):
        x_1 = x[j][:, :913]
        y_1 = y[j][0, 1000: 1913]
        # y_test = y[j][0, 1913:]
        m = sm.OLS(y_1, x_1.T).fit()
        y_va = m.predict(x[j][:, 913:941].T)
        y_ev = m.predict(x[j][:, 941:].T)
        y_v.append(y_va)
        y_e.append(y_ev)
        yv = np.array(y_v)
        ye = np.array(y_e)
        y_pre = np.array(pd.concat([pd.DataFrame(yv), pd.DataFrame(ye)]))
        # r_all.append(r_oos(y_test, y_pre))
        
        # r_adj = []
        # for i in range(x_1.shape[1] - 500):
        #     xx = x_1[:, i : i + 500]
        #     yy = y_1[i : i + 500]
        #     m = sm.OLS(yy, sm.add_constant(xx.T)).fit()
        #     r_adj.append(m.rsquared_adj)
        # r_all.append(pd.DataFrame(r_adj).dropna().reset_index(drop = True).mean())
    ###################### lasso ###################
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha = 0.1)
    y_v = []
    y_e = []
    for j in tqdm(range(len(x))):
        x_1 = x[j][:, :913]
        y_1 = y[j][0, 1000: 1913]
        # y_test = y[j][0, 1913:]
        m = lasso.fit(x_1.T, y_1)
        y_va = m.predict(x[j][:, 913:941].T)
        y_ev = m.predict(x[j][:, 941:].T)
        y_v.append(y_va)
        y_e.append(y_ev)
        yv = np.array(y_v)
        ye = np.array(y_e)
        y_pre = np.array(pd.concat([pd.DataFrame(yv), pd.DataFrame(ye)]))
    
    ########################### knn ######################
    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors = 5)
    y_v = []
    y_e = []
    for j in tqdm(range(len(x))):
        x_1 = x[j][:, :913]
        y_1 = y[j][0, 1000: 1913]
        # y_test = y[j][0, 1913:]
        m = knn.fit(x_1.T, y_1)
        y_va = m.predict(x[j][:, 913:941].T)
        y_ev = m.predict(x[j][:, 941:].T)
        y_v.append(y_va)
        y_e.append(y_ev)
        yv = np.array(y_v)
        ye = np.array(y_e)
        y_pre = np.array(pd.concat([pd.DataFrame(yv), pd.DataFrame(ye)]))
        
    ############################ svm #########################
    from sklearn.svm import SVR
    svr = SVR()
    y_v = []
    y_e = []
    for j in tqdm(range(len(x))):
        x_1 = x[j][:, :913]
        y_1 = y[j][0, 1000: 1913]
        # y_test = y[j][0, 1913:]
        m = svr.fit(x_1.T, y_1)
        y_va = m.predict(x[j][:, 913:941].T)
        y_ev = m.predict(x[j][:, 941:].T)
        y_v.append(y_va)
        y_e.append(y_ev)
        yv = np.array(y_v)
        ye = np.array(y_e)
        y_pre = np.array(pd.concat([pd.DataFrame(yv), pd.DataFrame(ye)]))
        
    ######################## RF ########################
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators= 50)
    y_v = []
    y_e = []
    for j in tqdm(range(len(x))):
        x_1 = x[j][:, :913]
        y_1 = y[j][0, 1000: 1913]
        # y_test = y[j][0, 1913:]
        m = rf.fit(x_1.T, y_1)
        y_va = m.predict(x[j][:, 913:941].T)
        y_ev = m.predict(x[j][:, 941:].T)
        y_v.append(y_va)
        y_e.append(y_ev)
        yv = np.array(y_v)
        ye = np.array(y_e)
        y_pre = np.array(pd.concat([pd.DataFrame(yv), pd.DataFrame(ye)]))
        
        
    ######################### GDBT ##########################
    from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
    from lightgbm import LGBMRegressor
    gdbt = GradientBoostingRegressor()
    gdbt = LGBMRegressor()
    y_v = []
    y_e = []
    for j in tqdm(range(25000, 30490)):
        x_1 = x[j][:, 741:941]
        y_1 = y[j][0, 1741: 1941]
        # y_test = y[j][0, 1913:]
        m = gdbt.fit(x_1.T, y_1)
        y_va = m.predict(x[j][:, 913:941].T)
        y_ev = m.predict(x[j][:, 941:].T)
        y_v.append(y_va)
        y_e.append(y_ev)
    yv = np.array(y_v)
    ye = np.array(y_e)
    y_pre = np.array(pd.concat([pd.DataFrame(yv), pd.DataFrame(ye)]))
        
        
    ##################### knn better =#########
    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors = 8)
    y_v = []
    y_e = []
    for j in tqdm(range(len(x))):
        x_1 = x[j][:, 863:913]
        y_1 = y[j][0, 1863: 1913]
        x_2 = x[j][:, 813:863]
        y_2 = y[j][0, 1813: 1863]
        x_3 = x[j][:, 763:813]
        y_3 = y[j][0, 1763: 1813]
        # x_4 = x[j][:, 833:853]
        # y_4 = y[j][0, 1833: 1853]
        # x_5 = x[j][:, 813:833]
        # y_5 = y[j][0, 1813: 1833]
        # y_test = y[j][0, 1913:]
        m1 = knn.fit(x_1.T, y_1)
        m2 = knn.fit(x_2.T, y_2)
        m3 = knn.fit(x_3.T, y_3)
        # m4 = knn.fit(x_4.T, y_4)
        # m5 = knn.fit(x_5.T, y_5)
        y_va = (m1.predict(x[j][:, 913:941].T) + m2.predict(x[j][:, 913:941].T) + m3.predict(x[j][:, 913:941].T)) / 3
        y_ev = (m1.predict(x[j][:, 941:].T) + m2.predict(x[j][:, 941:].T) + m3.predict(x[j][:, 941:].T)) / 3
        y_v.append(y_va)
        y_e.append(y_ev)
        yv = np.array(y_v)
        ye = np.array(y_e)
        y_pre = np.array(pd.concat([pd.DataFrame(yv), pd.DataFrame(ye)]))
        
        
    ######################## knn + lr ##########################
    from sklearn.neighbors import KNeighborsRegressor
    import statsmodels.api as sm
    from sklearn.linear_model import Lasso
    from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
    from xgboost import XGBRFRegressor
    gdbt = GradientBoostingRegressor(min_samples_split= 10, min_samples_leaf = 100, max_depth = 5)
    knn = KNeighborsRegressor(n_neighbors = 5)
    xgb = XGBRFRegressor(max_depth=5, learning_rate=0.1, n_estimators=160)
    lasso = Lasso(alpha = 0.1)
    y_v = []
    y_e = []
    for j in tqdm(range(20000, 30490)):
        x_1 = x[j][:, 741:941]
        y_1 = y[j][0, 1741: 1941]
        x_2 = x[j][:, 841:891]
        y_2 = y[j][0, 1841: 1891]
        x_3 = x[j][:, 791:841]
        y_3 = y[j][0, 1791: 1841]
        # x_4 = x[j][:, 741:791]
        # y_4 = y[j][0, 1741: 1791]
        # x_5 = x[j][:, 691:741]
        # y_5 = y[j][0, 1691: 1741]
        y_test = y[j][0, 1913:]
        m1 = gdbt.fit(x_1.T, y_1)
        m2 = xgb.fit(x_2.T, y_2)
        m3 = xgb.fit(x_3.T, y_3)
        # m1k = lasso.fit(x_1.T, y_1)
        # m2k = lasso.fit(x_2.T, y_2)
        # m3k = lasso.fit(x_3.T, y_3)
        
        
        # m4 = gdbt.fit(x_4.T, y_4)
        # m5 = knn.fit(x_5.T, y_5)
        
        y1_t = m1.predict(x[j][:, 913:941].T)
        y2_t = m2.predict(x[j][:, 913:941].T)
        y3_t = m3.predict(x[j][:, 913:941].T)
        # y1_tk = m1k.predict(x[j][:, 913:941].T)
        # y2_tk = m2k.predict(x[j][:, 913:941].T)
        # y3_tk = m3k.predict(x[j][:, 913:941].T)
        
        
        # y4_t = m4.predict(x[j][:, 913:941].T)
        # y5_t = m5.predict(x[j][:, 913:941].T)
        
        yy = pd.concat([pd.DataFrame(y1_t), pd.DataFrame(y2_t), pd.DataFrame(y3_t)], axis = 1)
        m_r = sm.OLS(y_test, yy).fit()
        y_va = y_test + np.array(m_r.resid)
        
        
        y1_v = m1.predict(x[j][:, 941:].T)
        y2_v = m2.predict(x[j][:, 941:].T)
        y3_v = m3.predict(x[j][:, 941:].T)
        # y1_vk = m1k.predict(x[j][:, 941:].T)
        # y2_vk = m2k.predict(x[j][:, 941:].T)
        # y3_vk = m3k.predict(x[j][:, 941:].T)
        
        
        # y4_v = m4.predict(x[j][:, 941:].T)
        # y5_v = m5.predict(x[j][:, 941:].T)
        yy_v = pd.concat([pd.DataFrame(y1_v), pd.DataFrame(y2_v), pd.DataFrame(y3_v)], axis = 1)
        
        y_ev = m_r.predict(yy_v)
        
        # y_va = (m1.predict(x[j][:, 913:941].T) + m2.predict(x[j][:, 913:941].T) + m3.predict(x[j][:, 913:941].T)) / 3
        # y_ev = (m1.predict(x[j][:, 941:].T) + m2.predict(x[j][:, 941:].T) + m3.predict(x[j][:, 941:].T)) / 3
        y_v.append(y_va)
        y_e.append(y_ev)
        
        
        
        yv = np.array(y_v)
        ye = np.array(y_e)
        y_pre = np.array(pd.concat([pd.DataFrame(yv), pd.DataFrame(ye)]))
        
    ################### NN and lightgbm ################
    from tqdm import trange
    from sklearn.preprocessing import StandardScaler 
    from sklearn.neural_network import MLPRegressor
    d_raw = pd.read_csv(r'D:\M5 forcasting\data_final_c.csv')

    # train,test set split and data propocessing
    lst=[]
    for i in d_raw['d']:
      lst.append(i.split('_')[1])
      
    d_raw['day']=lst
    d_raw['day']=d_raw['day'].astype(np.int16)
    
    def rmse(predictions, targets):
        """calculate root mean squared error"""
        return np.sqrt(((predictions - targets) ** 2).mean())
    
    data = d_raw[d_raw['day']<1914].set_index("id")
    train = data[data['day']>=1800]  # original 1180
    pre = d_raw[d_raw['day']>=1914].set_index("id").drop(['d','day','demand'], axis=1)
    trainLabel = train["demand"]
    train.drop(['d','day','demand'],axis=1,inplace=True)
    
    test = data[(data['day']>=1700) & (data['day']<1800)]   # 180 days for test
    testLabel = test["demand"]
    test.drop(['d','day','demand'],axis=1,inplace=True)
    
    grouped_data=list(data.groupby(['id']))
    
    # train a model for each id
    df1 = pd.DataFrame(index=list(set(data.index)),columns=[i for i in range(1,57)])
    df2 = pd.DataFrame(index=list(set(data.index)),columns=[i for i in range(1,57)])
    df3 = pd.DataFrame(index=list(set(data.index)),columns=[i for i in range(1,57)])
    rmse_lst1 = []
    rmse_lst2 = []
    for i in trange(0,len(grouped_data),50): #13208
        sdata = grouped_data[i][1]  
        pre1 = pre[pre.index==grouped_data[i][0]]
        nn2 = MLPRegressor(hidden_layer_sizes=(16,8),max_iter=650).fit(train.loc[grouped_data[i][0]],trainLabel.loc[grouped_data[i][0]])
        y_hat1 = nn2.predict(pre1)
        nn4 = MLPRegressor(hidden_layer_sizes=(20,16,8,6),max_iter=900).fit(train.loc[grouped_data[i][0]],trainLabel.loc[grouped_data[i][0]])
        y_hat2 = nn4.predict(pre1)
        gbm = LGBMRegressor().fit(train.loc[grouped_data[i][0]],trainLabel.loc[grouped_data[i][0]])
        y_hat3 = gbm.predict(pre1)
        # rmse_score1 = rmse(y_hat1,testLabel.loc[grouped_data[i][0]])
        # rmse_score2 = rmse(y_hat2,testLabel.loc[grouped_data[i][0]])
    
        df1.loc[grouped_data[i][0]]= y_hat1
        df2.loc[grouped_data[i][0]]= y_hat2
        df3.loc[grouped_data[i][0]]= y_hat3
        

    
