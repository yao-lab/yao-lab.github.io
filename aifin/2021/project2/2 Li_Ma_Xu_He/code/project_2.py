# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 15:09:06 2021

@author: xtcxxx
"""

########### data processing ###############
import pandas as pd
df = pd.read_csv('GKX_20201231.csv')

#%%
firm_char=['absacc', 'acc', 'aeavol', 'age', 'agr', 'baspread', 'beta', 
             'betasq', 'bm', 'bm_ia', 'cash', 'cashdebt', 'cashpr',  
             'cfp', 'cfp_ia', 'chatoia', 'chcsho', 
             'chempia', 'chinv', 'chmom',  'chpmia', 'chtx', 
             'cinvest', 'convind', 
             'currat', 'depr', 'divi', 'divo', 'dolvol', 'dy', 
             'ear', 'egr', 'ep', 'gma', 'grcapx',  'grltnoa', 'herf', 
             'hire', 'idiovol', 'ill', 'indmom', 'invest',  'lev', 'lgr', 
             'maxret', 'mom12m', 'mom1m', 'mom36m', 'mom6m', 'ms', 'mvel1', 
             'mve_ia', 'nincr', 'operprof', 'orgcap', 
             'pchcapx_ia', 'pchcurrat', 'pchdepr', 'pchgm_pchsale', 'pchquick', 
             'pchsale_pchinvt', 'pchsale_pchrect', 'pchsale_pchxsga', 
             'pchsaleinv', 'pctacc', 'pricedelay', 'ps', 'quick', 
             'rd', 'rd_mve', 'rd_sale', 'realestate', 
             'retvol', 'roaq', 'roavol', 
             'roeq', 'roic', 'rsup', 'salecash', 'saleinv', 'salerec', 
             'secured', 'securedind', 'sgr', 'sin', 'sp',  
             'std_dolvol', 'std_turn', 'stdacc', 'stdcf', 'tang', 'tb', 
             'turn', 'zerotrade']
sec = ['permno', 'DATE', 'RET', 'prc', 'SHROUT','mve0','sic2']
import numpy as np
df_cols = np.append(np.asarray(sec), firm_char)
df = df[df_cols]
macro = pd.read_excel('PredictorData2020.xlsx', parse_dates=['yyyymm'])
df1 = df.loc[(df['DATE']>=19570101)&(df['DATE']<=20161231)]
film_char1=['permno','DATE','RET','prc','SHROUT','mve0','sic2','baspread','beta','betasq','chmom',
           'dolvol','idiovol','ill','indmom','maxret','mom12m','mom1m','mom6m',
           'mvel1','pricedelay','retvol','std_dolvol','std_turn','turn','zerotrade']
df3=df1[film_char1]
df3 = df3.fillna(0)
df3.to_csv('D:/6010/datashare/data1.csv',index=False)
#%%
macro['dp'] = np.log(macro['D12']) - np.log(macro['Index'])
macro['ep'] = np.log(macro['E12']) - np.log(macro['Index'])
macro['bm'] = macro['b/m']
macro['tms'] = macro['lty'] - macro['tbl']
macro['dfy'] = macro['BAA'] - macro['AAA']
macro1 = macro.loc[(macro['yyyymm']>=195701)&(macro['yyyymm']<=201612),
                   ['yyyymm', 'dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar']]

num=df1.isna().sum()
macro1.to_csv('D:/6010/datashare/macro.csv',index=False)
#%%
import pandas as pd
df_firm = pd.read_csv('data1.csv')
macro = pd.read_csv('macro.csv')
df_firm['yyyymm'] = df_firm['DATE'].astype(str).str[:6].astype(int)
data = pd.merge(df_firm, macro, how='left', on='yyyymm', suffixes=('', '_macro'))
firm_char1=['baspread','beta','betasq','chmom',
           'dolvol','idiovol','ill','indmom','maxret','mom12m','mom1m','mom6m',
           'mvel1','pricedelay','retvol','std_dolvol','std_turn','turn','zerotrade']
macro_cols = ['dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar']
a = data[firm_char1].to_numpy(copy=False)
b = data[macro_cols].to_numpy(copy=False)
import numexpr as ne
df_firm['yyyymm'] = df_firm['DATE'].astype(str).str[:6].astype(int)
data = pd.merge(df_firm, macro, how='left', on='yyyymm', suffixes=('', '_macro'))
out = ne.evaluate('a3D*b3D',{'a3D':a[:,:,None],'b3D':b[:,None]}).reshape(len(a),-1)
df_out = pd.DataFrame(out)
df_out.columns = [f"{i}*{j}" for i in firm_char1 for j in macro_cols]
df_out.index = data.index

import numpy as np
from functools import reduce
sec = ['permno', 'DATE', 'RET', 'prc', 'SHROUT','mve0','sic2']
cols =[np.asarray(sec), firm_char1, df_out.columns.values]
data_cols = reduce(lambda left, right: np.append(left, right), cols)
data_new = data[data_cols]
data_new.to_csv('D:/6010/datashare/data_new.csv',index=False)

############## OLS and LASSO ######################
import pandas as pd
import numpy as np
from tqdm import tqdm
df=pd.read_csv(r'D:\Learning\HKUST\6010Z\project2\data_new.csv')

#recursive
PLS_score=[]
ENet_score=[]
df['Year'] = df['DATE'].apply(lambda x: int(x / 10000))
for i in tqdm(range(30)):
    df_train = (df.loc[df["Year"]<=1974+i]).iloc[:,:-1]
    df_val = (df.loc[(df["Year"]>=1975+i)&(df["Year"]<=1986+i)]).iloc[:,:-1]
    df_test = (df.loc[df["Year"]==1987+i]).iloc[:,:-1]
    x_train = df_train.drop(['RET','DATE','prc','SHROUT','mve0'],axis=1)
    y_train = df_train["RET"]
    x_val = df_val.drop(['RET','DATE','prc','SHROUT','mve0'],axis=1)
    y_val = df_val["RET"]
    x_test = df_test.drop(['RET','DATE','prc','SHROUT','mve0'],axis=1)
    y_test = df_test["RET"]
    
    from sklearn.preprocessing import StandardScaler
    stdsc = StandardScaler()
    x_train_std = stdsc.fit_transform(x_train)
    x_val_std = stdsc.transform(x_val)
    x_test_std = stdsc.transform(x_test)


    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    lr = LinearRegression()
    lr.fit(x_train_std, y_train.astype('int'))
    y_pred1 = lr.predict(x_test_std)
    print('OLS accuracy:', r2_score(y_test,y_pred1))
    PLS_score.append(r2_score(y_test,y_pred1))

    from sklearn.linear_model import ElasticNet
    ENreg = ElasticNet(alpha=1, l1_ratio=0)
    ENreg.fit(x_train_std, y_train.astype('int'))
    y_pred2 = ENreg.predict(x_test_std)
    print('ElasticNet accuracy:', r2_score(y_test,y_pred2))
    ENet_score.append(r2_score(y_test,y_pred2))


'''
data_train = df.loc[(df['DATE']>=19570101) & (df['DATE']<=19741231)]
x_train = data_train.drop(['RET','DATE','prc','SHROUT','mve0'],axis=1)
y_train = data_train["RET"]

data_val = df.loc[(df['DATE']>=19750101) & (df['DATE']<=19861231)]
x_val = data_val.drop(['RET','DATE','prc','SHROUT','mve0'],axis=1)
y_val = data_val["RET"]

data_test = df.loc[(df['DATE']>=19870101) & (df['DATE']<=19871231)]
x_test = data_test.drop(['RET','DATE','prc','SHROUT','mve0'],axis=1)
y_test = data_test["RET"]

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
x_train_std = stdsc.fit_transform(x_train)
x_val_std = stdsc.transform(x_val)
x_test_std = stdsc.transform(x_test)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
lr = LinearRegression()
lr.fit(x_train_std, y_train.astype('int'))
y_pred1 = lr.predict(x_test_std)
print('OLS accuracy:', r2_score(y_test,y_pred1))
importance_OLS = lr.coef_.tolist()
feature = list(x_test)
df1=pd.DataFrame({'feature':feature,'importance':importance_OLS})
df1.sort_values("importance",inplace=True,ascending=True,ignore_index=True)
df1=df1.tail(20)
df1.plot(kind = 'barh', x = 'feature', color = 'blue', title = 'OLS')



from sklearn.linear_model import ElasticNet
ENreg = ElasticNet(alpha=1, l1_ratio=0)
ENreg.fit(x_train_std, y_train.astype('int'))
y_pred2 = ENreg.predict(x_test_std)
print('ElasticNet accuracy:', r2_score(y_test,y_pred2))
importance_ENet = ENreg.fit(x_train_std, y_train.astype('int')).coef_
feature = list(x_test)
df2=pd.DataFrame({'feature':feature,'importance':importance_ENet})
df2.sort_values("importance",inplace=True,ascending=True,ignore_index=True)
df2=df2.tail(20)
df2.plot(kind = 'barh', x = 'feature', color = 'blue', title = 'ENet')
'''

############### PCR #########################
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from pandas import DataFrame
import matplotlib.pyplot as plt
data1 = pd.read_csv(r'C:\Users\Administrator\Desktop\data_new.csv')
dnew = data1.drop(['permno', 'RET'], axis = 1)
d_new = dnew[dnew['DATE'] < 19750101].reset_index(drop = True)
d_new = d_new.drop(['DATE'], axis = 1)

d_size = data1[['permno', 'mvel1']]
d_size.sort_values(by = 'mvel1', ascending = False)
top_firm = d_size['permno'].iloc[:1000]
bottom_firm = d_size['permno'].iloc[-1000:]
d_top = data1[data1['permno'] == list(top_firm)]

def filter_size(data, company_name):
    result = pd.DataFrame([])
    for f in company_name:
        d1 = data[data['permno'] == f]
        result = pd.concat([result, d1])
    result = result.sort_values(by = 'DATE').reset_index(drop = True)
    return result

kk = filter_size(data1, list(bottom_firm))


def calculate_r2(y,y_bar):
    sst = np.sum((y - np.mean(y))**2)
    sse = np.sum((y - y_bar)**2)
    r2 = 1 - sse / sst
    return r2


d2 = data1[data1['DATE'] < 19750101].reset_index(drop = True)
x = d2.drop(['permno', 'RET', 'DATE'], axis = 1)
x_0 = x.copy()
x_0.iloc[:,:] = 0
pca_0 = PCA(n_components= 1)
pca_0.fit(x)
predictors = x.columns
predictors = list(predictors)
y = d2['RET']
x_n = pca_0.transform(x)
model_0 = sm.OLS(y, x_n)
m0 = model_0.fit()
coeff = m0.params
coeff = np.array(coeff)
r2 = []
for p in predictors:
    x1 = np.array(x[p])
    x_n = pca_0.transform(x1.reshape(-1, 1))
    model_0 = sm.OLS(y, x_n)
    m0 = model_0.fit()
    r2.append(m0.rsquared)

r2 = pd.DataFrame(r2)
rr = pd.concat([pd.DataFrame(predictors), r2], axis = 1)
rr.columns = ['feature', 'importance']
rr.sort_values(by = 'importance', ascending= False)
rr = rr.sort_values(by = 'importance', ascending= False)

rr = rr.iloc[:20,:].sort_values(by = 'importance')
rr['importance'] = (rr['importance'] - np.mean(rr['importance'])) / np.std(rr['importance'])
rr['importance'] = rr['importance'].abs() / 10
rr = rr.sort_values(by = 'importance')
rr.plot(kind = 'barh', x = 'feature', color = 'blue', title = 'PCR')



pca1 = PCA(n_components= d_new.shape[1])
pca1.fit(d_new)

def pcr(data1: DataFrame):
    score = []
    r2 = []
    for i in range(1, 30):
        date = 19740101 + i * 10000
        d_train = data1[data1['DATE'] < date].reset_index(drop = True)
        d_v = data1[(data1['DATE'] >= date) & (data1['DATE'] < date + (12+i)*10000)].reset_index(drop = True)
        d_test = data1[(data1['DATE'] >= date + (12+i)*10000) & (data1['DATE'] < date + (13+i)*10000)].reset_index(drop = True)
        d_t1 = d_train.drop(['permno', 'DATE'], axis = 1)
        x = d_t1.drop(['RET'], axis = 1)
        x_v = d_v.drop(['permno', 'DATE', 'RET'], axis = 1)
        x_test = d_test.drop(['permno', 'DATE', 'RET'], axis = 1)
        pca = PCA(n_components= 10)
        pca.fit(x)
        x_n = pca.transform(x)
        x_v_n = pca.transform(x_v)
        x_test_n = pca.transform(x_test)
        y = d_train['RET']
        y_v = d_v['RET']
        y_test = d_test['RET']
        model = sm.OLS(y, x_n)
        pcr = model.fit()
        s = cross_val_score(model, x_v_n, y_v, cv = 4)
        score.append(s.mean())
        y_pre = pcr.predict(x_test_n)
        r2.append(calculate_r2(y_test, y_pre))
    return np.mean(r2)
if __name__ == '__main__':
    data_top = filter_size(data1, list(top_firm))
    data_bottom = filter_size(data1, list(bottom_firm))
    r_top = pcr(data_top)
    r_bottom = pcr(data_bottom)
    r_all = pcr(data1)
    
    
################## GDBT and RF ####################
import pandas as pd
data_new = pd.read_csv('data_new.csv')
#%%
data_new = data_new.join(pd.get_dummies(data_new.sic2))
#%%
del data_new["sic2"]
#%%
data_new=pd.get_dummies(data_new["sic2"])
d1=data_new["sic2"]
#%%
data_train = data_new.loc[(data_new['DATE']>=19570101)&(data_new['DATE']<=19741231)]
#%%
data_val = data_new.loc[(data_new['DATE']>=19750101)&(data_new['DATE']<=19861231)]
data_test = data_new.loc[(data_new['DATE']>=19870101)&(data_new['DATE']<=20161231)]

#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
x_train = data_train.drop(["RET","DATE"],axis=1)
y_train = data_train["RET"]
#%%
'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

n_estimators =  [130, 180, 230]
max_features = ['auto', 'sqrt']
max_depth = [1,2,3,4,5]
min_samples_split = [2, 6, 10]
min_samples_leaf = [1, 3, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
'max_features': max_features,
'max_depth': max_depth,
'min_samples_split': min_samples_split,
'min_samples_leaf': min_samples_leaf,
'bootstrap': bootstrap}
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
               n_iter = 100, cv = 5, verbose=2, random_state=35)

rf_random.fit(x_train, y_train)

from sklearn.model_selection import GridSearchCV
param_test1 = {'n_estimators':range(30,231,50)}
gsearch1 = GridSearchCV(estimator = RandomForestRegressor(min_samples_split=100,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10,n_jobs=-1), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(x_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#%%
y_val=data_val["RET"]
x_val=data_val.drop(["RET","DATE","prc","SHROUT","mve0"],axis=1)
#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
m=np.arange(5,106,20)
for i in np.arange(20,521,100):
    rf = RandomForestRegressor(n_estimators=80, max_depth=11,
                                   min_samples_leaf=65,
                                   min_samples_split=i,n_jobs=-1)
    rf.fit(x_train,y_train)
    
    y_pred1=rf.predict(x_val)
    print(r2_score(y_val,y_pred1))
'''    

#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
r2=pd.DataFrame()
for i in np.arange(1,31,1):
    data_train = data_new.loc[(data_new['DATE']>=19570101)&(data_new['DATE']<=(19741231+(i-1)*10000))]
    x_train = data_train.drop(["RET","DATE","prc","SHROUT","mve0"],axis=1)
    y_train = data_train["RET"]
    rf = RandomForestRegressor(n_estimators=80, max_depth=11,
                                   min_samples_leaf=65,
                                   min_samples_split=1700,n_jobs=-1,oob_score=True)
    rf.fit(x_train,y_train)
    data_test = data_new.loc[(data_new['DATE']>=(19870101+(i-1)*10000))&(data_new['DATE']<=(19871231+(i-1)*10000))]
    y_test=data_test["RET"]
    x_test=data_test.drop(["RET","DATE","prc","SHROUT","mve0"],axis=1)
    y_pred=rf.predict(x_test)
    r2[i]=[r2_score(y_test,y_pred)]
    print(r2[i])
#%%
print(r2_score(y_test,y_pred))
#%%
tree_feature_importances = rf.feature_importances_
feature_names=np.r_[x_train.columns.values]

#%%
import numpy as np
d2=tree_feature_importances.T
d3=feature_names.T
d1=pd.DataFrame(np.vstack((d3,d2)))
d1=d1.T
names=['feature','importance']
d1.columns=names
#d1.set_index('feature',inplace=True)
d4=d1.sort_values(by=['importance'],ascending=False)
d4.reset_index(drop=True, inplace=True)
#%%
d5=d4.loc[1:20]
#%%

d6=d5.sort_values(by=['importance'])
import matplotlib.pyplot as plot
d6.plot(kind = 'barh', x = 'feature', color = 'blue', title = 'Random Forest')
#%%
import pandas as pd
data_bottom = pd.read_csv('data_bottom.csv')
#%%
from sklearn.metrics import r2_score
r21=pd.DataFrame()
for i in np.arange(1,31,1):
    data_test = data_bottom.loc[(data_bottom['DATE']>=(19870101+(i-1)*10000))&(data_bottom['DATE']<=(19871231+(i-1)*10000))]
    y_test=data_test["RET"]
    x_test=data_test.drop(["RET","DATE","prc","SHROUT","mve0"],axis=1)
    y_pred=rf.predict(x_test)
    r21[i]=[r2_score(y_test,y_pred)]
    print(r21[i])
#%%
r211=r21.T
#%%
import pandas as pd

data_top = pd.read_csv('data_top1.csv')

#%%
from sklearn.metrics import r2_score
r22=pd.DataFrame()
for i in np.arange(1,31,1):
    data_test = data_top.loc[(data_top['DATE']>=(19870101+(i-1)*10000))&(data_top['DATE']<=(19871231+(i-1)*10000))]
    y_test=data_test["RET"]
    x_test=data_test.drop(["RET","DATE","prc","SHROUT","mve0"],axis=1)
    y_pred=rf.predict(x_test)
    r22[i]=[r2_score(y_test,y_pred)]
    print(r22[i])
#%%
r222=r22.T

import pandas as pd
data_new = pd.read_csv('data_new.csv')
#%%
'''
data_train = data_new.loc[(data_new['DATE']>=19570101)&(data_new['DATE']<=19741231)]
data_val = data_new.loc[(data_new['DATE']>=19750101)&(data_new['DATE']<=19861231)]
data_test = data_new.loc[(data_new['DATE']>=19870101)&(data_new['DATE']<=20161231)]
'''
#%%
'''
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
x_train = data_train.drop(["RET","DATE","prc","SHROUT","mve0"],axis=1)
y_train = data_train["RET"]
y_val=data_val["RET"]
x_val=data_val.drop(["RET","DATE","prc","SHROUT","mve0"],axis=1)
gdbt = GradientBoostingRegressor(learning_rate= 0.1, min_samples_split= 5000, min_samples_leaf= 5,
                                      max_depth= 11 )
gdbt.fit(x_train, y_train)
y_pred1 = gdbt.predict(x_val)
print(r2_score(y_val,y_pred1))
'''
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import numpy as np
r2=pd.DataFrame()
for i in np.arange(1,31,1):
    data_train = data_new.loc[(data_new['DATE']>=19570101)&(data_new['DATE']<=(19741231+(i-1)*10000))]
    x_train = data_train.drop(["RET","DATE","prc","SHROUT","mve0"],axis=1)
    y_train = data_train["RET"]
    gdbt = GradientBoostingRegressor(learning_rate= 0.1, min_samples_split= 5000, min_samples_leaf= 5,
                                      max_depth= 11 )
    gdbt.fit(x_train, y_train)
    data_test = data_new.loc[(data_new['DATE']>=(19870101+(i-1)*10000))&(data_new['DATE']<=(19871231+(i-1)*10000))]
    y_test=data_test["RET"]
    x_test=data_test.drop(["RET","DATE","prc","SHROUT","mve0"],axis=1)
    y_pred=gdbt.predict(x_test)
    r2[i]=[r2_score(y_test,y_pred)]
    print(r2[i])
#%%
import numpy as np
tree_feature_importances = gdbt.feature_importances_
feature_names=np.r_[x_train.columns.values]

#%%
import numpy as np
d2=tree_feature_importances.T
d3=feature_names.T
d1=pd.DataFrame(np.vstack((d3,d2)))
d1=d1.T
names=['feature','importance']
d1.columns=names
#d1.set_index('feature',inplace=True)
d4=d1.sort_values(by=['importance'],ascending=False)
d4.reset_index(drop=True, inplace=True)
#%%
d5=d4.loc[1:20]
d6=d5.sort_values(by=['importance'])
import matplotlib.pyplot as plot
d6.plot(kind = 'barh', x = 'feature', color = 'blue', title = 'GDBT')


############ NN ######################
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv(r"D:\data_new.csv")
top_s = pd.read_csv(r"D:\data_top.csv")
bottom_s = pd.read_csv(r"D:\data_bottom.csv")

nn1_scores = []
nn2_scores = []
nn3_scores = []
nn4_scores = []
nn5_scores = []


for i in range(0,14,29):
    train = data[(data["DATE"]>=19570130) & (data["DATE"]<=19741231+i*10000)]  # 18 years  #####
    #validate = data[(data["DATE"]>=19750130+i*10000) & (data["DATE"]<=19861231+i*10000)]  # 12 years] 
    test = data[(data["DATE"]>=19570130+(i+30)*10000) & (data["DATE"]<=19571231+(i+30)*10000)]  # 1 years  ####
    top = top_s[(top_s["DATE"]>=19570130) & (top_s["DATE"]<=19741231+i*10000)]
    top_test = top_s[(top_s["DATE"]>=19570130+(i+30)*10000) & (top_s["DATE"]<=19571231+(i+30)*10000)]  ####
    bottom = bottom_s[(bottom_s["DATE"]>=19840130) & (bottom_s["DATE"]<=20011231)]
    bottom_test = bottom_s[(bottom_s["DATE"]>=20120130) & (bottom_s["DATE"]<=20121231)] ####



    trainLabel = train["RET"]
    train = train.drop(["permno","RET","DATE"],axis=1)
    #print(train.describe()) 
    topLabel = top["RET"]


    bottomLabel = bottom["RET"]


    for col in train.columns:
        train[col].fillna(0,inplace=True)

    testLabel = test["RET"]
    test = test.drop(["permno","RET","DATE"],axis=1)
    for col in test.columns:
        test[col].fillna(0,inplace=True)
    print(len(testLabel))

    topLabel_test = top_test["RET"]
    top_test = top_test.drop(["permno","RET","DATE"],axis=1)


    bottomLabel_test = bottom_test["RET"]
    bottom_test = bottom_test.drop(["permno","RET","DATE"],axis=1)

    top = top.drop(["permno","RET","DATE"],axis=1)
    bottom = bottom.drop(["permno","RET","DATE"],axis=1)

    #     validateLabel = validate["RET"]
    #     validate = validate.drop(["permno","RET","DATE"],axis=1)
    #     for col in validate.columns:
    #         validate[col].fillna(0,inplace=True)        
    print("right")
    scaler = StandardScaler()  
    # Don't cheat - fit only on training data
    scaler.fit(train)
    X_train = scaler.transform(train)  
    X_train = pd.DataFrame(X_train,columns=train.columns)
                 
    # apply same transformation to test data
    #     X_test = scaler.transform(test)
    #     X_test = pd.DataFrame(X_test,columns=test.columns)
    #     X_test = pd.concat([X_test,test_dummies],axis=1)
    X_test=scaler.transform(test)
    X_test=pd.DataFrame(X_test,columns=test.columns)
                     
    X_top=scaler.transform(top)
    X_top=pd.DataFrame(X_top,columns=top.columns)
                     
    X_bottom=scaler.transform(bottom)
    X_bottom=pd.DataFrame(X_bottom,columns=bottom.columns)
    
    X_bottom_test=scaler.transform(bottom_test)
    X_bottom_test=pd.DataFrame(X_bottom_test,columns=bottom_test.columns)
    
    X_top_test=scaler.transform(top_test)
    X_top_test=pd.DataFrame(X_top_test,columns=bottom.columns)
                     
    nn1_score = []
    nn2_score = []
    nn3_score = []
    nn4_score = []
    nn5_score= []
    print("data cleaned")

    #     optimizers = ['lbfgs','sgd', 'adam']  #  
    #     activates = ["identity","logistic","tanh","relu"] 


    nn1 = MLPRegressor(hidden_layer_sizes=(32,),solver='sgd',max_iter=1000).fit(X_train,trainLabel)
    nn1_top = MLPRegressor(hidden_layer_sizes=(32,),solver='sgd',max_iter=1000).fit(X_top,topLabel)
    nn1_bottom = MLPRegressor(hidden_layer_sizes=(32,),solver='sgd',max_iter=1000).fit(X_bottom,bottomLabel)
    joblib.dump(nn1,f"D://{i+1}_nn1.pkl")
    print("nn1")
    nn1_score.append([nn1.score(X_test,testLabel),nn1_top.score(X_top_test,topLabel_test),nn1_bottom.score(X_bottom_test,bottomLabel_test)])
    print(nn1_score)

    nn2 = MLPRegressor(hidden_layer_sizes=(32,16),solver='sgd',max_iter=1000).fit(X_train,trainLabel)
    nn2_top = MLPRegressor(hidden_layer_sizes=(32,16),solver='sgd',max_iter=1000).fit(X_top,topLabel)
    nn2_bottom = MLPRegressor(hidden_layer_sizes=(32,16),solver='sgd',max_iter=1000).fit(X_bottom,bottomLabel)
    print("nn2")
    joblib.dump(nn1,f"D://{i+1}_nn2.pkl")
    nn2_score.append([nn2.score(X_test,testLabel),nn2_top.score(X_top_test,topLabel_test),nn2_bottom.score(X_bottom_test,bottomLabel_test)])
    print(nn2_score)

    nn3 = MLPRegressor(hidden_layer_sizes=(32,16,8),solver='sgd',max_iter=1500).fit(X_train,trainLabel)
    nn3_top = MLPRegressor(hidden_layer_sizes=(32,16,8),solver='sgd',max_iter=1500).fit(X_top,topLabel)
    nn3_bottom = MLPRegressor(hidden_layer_sizes=(32,16,8),solver='sgd',max_iter=1500).fit(X_bottom,bottomLabel)
    print("nn3")
    joblib.dump(nn3,f"D://{i+1}_nn3.pkl")
    nn3_score.append([nn3.score(X_test,testLabel),nn3_top.score(X_top_test,topLabel_test),nn3_bottom.score(X_bottom_test,bottomLabel_test)])
    print(nn3_score)

    nn4 = MLPRegressor(hidden_layer_sizes=(32,16,8,4),solver='sgd',max_iter=2000,learning_rate="adaptive").fit(X_train,trainLabel)
    nn4_top = MLPRegressor(hidden_layer_sizes=(32,16,8,4),solver='sgd',max_iter=2000,learning_rate="adaptive").fit(X_top,topLabel)
    nn4_bottom = MLPRegressor(hidden_layer_sizes=(32,16,8,4),solver='sgd',max_iter=2000,learning_rate="adaptive").fit(X_bottom,bottomLabel)
    print("nn4")
    joblib.dump(nn4,f"D://{i+1}_nn4.pkl")
    nn4_score.append([nn4.score(X_test,testLabel),nn4_top.score(X_top_test,topLabel_test),nn4_bottom.score(X_bottom_test,bottomLabel_test)])
    print(nn4_score)

    nn5 = MLPRegressor(hidden_layer_sizes=(32,16,8,4,2),solver='sgd',max_iter=3000,learning_rate="adaptive").fit(X_train,trainLabel)
    nn5_top = MLPRegressor(hidden_layer_sizes=(32,16,8,4,2),solver='sgd',max_iter=3000,learning_rate="adaptive").fit(X_top,topLabel)
    nn5_bottom = MLPRegressor(hidden_layer_sizes=(32,16,8,4,2),solver='sgd',max_iter=3000,learning_rate="adaptive").fit(X_bottom,bottomLabel)
    joblib.dump(nn1,f"D://{i+1}_nn5.pkl")
    nn5_score.append([nn5.score(X_test,testLabel),nn5_top.score(X_top_test,topLabel_test),nn5_bottom.score(X_bottom_test,bottomLabel_test)])
    print("nn5")
    print(nn5_score)

    nn1_scores.append(nn1_score)
    nn2_scores.append(nn2_score)
    nn3_scores.append(nn3_score)
    nn4_scores.append(nn4_score)
    nn5_scores.append(nn5_score)
    print(f"round{i}")

# feature importance
feature_score_nn1 = []
feature_score_nn2 = []
feature_score_nn3 = []
feature_score_nn4 = []
feature_score_nn5 = []
for col in range(len(X_test.columns)):
    tt =X_test.copy()
    tt.iloc[:,col].fillna(0,inplace=True)
    tt.iloc[:,(col+1):] = 0
    tt.iloc[:,:col] = 0
    # nn1 = joblib.load()
    feature_score_nn1.append(nn1.score(tt,testLabel))
    feature_score_nn2.append(nn2.score(tt,testLabel))
    feature_score_nn3.append(nn3.score(tt,testLabel))
    feature_score_nn4.append(nn4.score(tt,testLabel))
    feature_score_nn5.append(nn5.score(tt,testLabel))
print(feature_score_nn1)
print(feature_score_nn2)
print(feature_score_nn3)
print(feature_score_nn4)
print(feature_score_nn5)
