import pandas as pd
import math

#from sklearn.linear_model import HuberRegressor
#import orginal data
data1 = pd.read_csv("C:/Users/yanxu/Desktop/6010Z/project2/GKX_20201231.csv")
data2 = pd.read_excel("C:/Users/yanxu/Desktop/6010Z/project2/PredictorData2020.xlsx")

#cut off the date range
#data3 = data1[data1.DATE>=19570131]
#data4 = data3[data3.DATE<=20161230]
#data4.to_csv('data.csv',index=False)
df = pd.read_csv("C:/Users/yanxu/Desktop/6010Z/project2/data.csv")
date = pd.DataFrame(df['DATE'].unique())
date = date.rename(columns = {0 : "DATE"})
#calculate the missing macroeco params
#fill NA by bfill method, fill in with the backward value
#this method is quite reasonable for macroparams,since it didnt fluctuate a lot
#another reason is the long time series of this large data, mean and median might be misleading
macrorefill = data2.fillna(method='bfill')
#by the macroparams definitions, get those params we neeed
f = lambda x: math.log(x)
data2['D12'] = macrorefill['D12'].apply(f)
data2['Index'] = macrorefill['Index'].apply(f)
data2['dp'] =macrorefill['D12'] - macrorefill['Index']
data2['tms'] = abs(macrorefill['lty'] - macrorefill['tbl'])
data2['dfy'] = abs(macrorefill['BAA'] - macrorefill['AAA'])

#pick out the output we need within the desired date range
macrooutput = data2[['yyyymm','dp','ntis','tbl','tms','dfy','svar']]
data5 = macrooutput[macrooutput.yyyymm>=195701]
macroparams = data5[data5.yyyymm<=201612]
#reset the index and join the suitable date form into the output
macroparams = macroparams.reset_index()
macroparams = macroparams.drop(['index'], axis = 1)
macroparams = macroparams.join(date)
macroparams = macroparams.drop(['yyyymm'], axis = 1)
#join the macroparms into the big data ,check&fix the NA terms
df1 = pd.merge(df,macroparams,on='DATE')
#progress the NA data with drop and fill function
df1.isnull().any().value_counts()

dropindex = df1[df1.sic2.isnull()].index.to_list()
df1 = df1.drop(index = dropindex)
df1  = df1.reset_index()
df1  = df1.drop(['index'], axis = 1)
#expand sic2 sector charteristic columns into dummy variable
df1 = df1.join(pd.get_dummies(df['sic2']))
#using median to refill the NA data
def medianfill(x):
    for i in x.columns[0:106]:
        x[i] = x[i].fillna(x[i].median())
    return x

#df2=df1.groupby(['sic2']).apply(lambda x: medianfill(x))
df2=df1.groupby(['permno']).apply(lambda x: medianfill(x))
#df2 = df1.copy()                   
#for i in df2.columns:
#    df2[i].fillna(df2[i].median(), inplace = True)
    
for i in df2.columns[0:107]:
    dropindex = df2[df2[i].isnull()].index.to_list()
    df2  = df2.drop(index = dropindex)
    df2  = df2.reset_index()
    df2  = df2.drop(['index'], axis = 1)


df2.isnull().any().value_counts()


#clear up those unused data for saving memory
def reset():
    global df
    del df
    global df1
    del df1
reset()

#specify the year of data
def get_yr(x):
   return int(x/10000)
df2['year']=df2['DATE'].apply(lambda x: get_yr(x))
year = pd.DataFrame(df2['year'].unique())
year = year.rename(columns = {0 : "year"})
#use those two function to define top/bot 1000 stocks based on its market equity
def get_top1000(x):
   df = x.sort_values(by = 'mvel1',ascending=False)
   return df.iloc[0:1000,:]

top1000=df2.groupby('DATE').apply(lambda x: get_top1000(x))

def get_bot1000(x):
   df = x.sort_values(by = 'mvel1',ascending=True)
   return df.iloc[0:1000,:]

bot1000=df2.groupby('DATE').apply(lambda x: get_bot1000(x))

##create a empty df for final output,save it and re-read it when we needed
total_output = pd.DataFrame(columns=['Model','All', 'top', 'bot'])

total_output.to_csv('total_output.csv',index=False)
total_output = pd.read_csv("C:/Users/yanxu/Desktop/6010Z/project2/total_output.csv")

##OLS-3
#recursive performance evaluation scheme
import statsmodels.api as sm
def recursive_eval_OLS3(df2):
    i_list = list()
    r2_list = list()
    for i in range(18, 48):
        #seprate the data
        trn = (year.iloc[0:i,:])['year'].to_list()
        train = df2[df2.year.isin(trn)]
        trn_x = train.drop(['RET','year'],axis=1)
        trn_y = train.RET
        
        vd = (year.iloc[i:i+12,:])['year'].to_list()
        valid = df2[df2.year.isin(vd)]
        valid_x = valid.drop(['RET','year'],axis=1)
        valid_y = valid.RET.reset_index(drop=True) 
              
        #huber = HuberRegressor().fit(trn_x[['mvel1','bm','mom12m']] ,trn_y)
        regression1 = sm.RLM(trn_y,trn_x[['mvel1','bm','mom12m']],M=sm.robust.norms.HuberT() ) 
        huber = regression1.fit()
        pred = pd.Series(huber.predict(valid_x[['mvel1','bm','mom12m']])).reset_index(drop=True)
        score = 1-sum(pow(valid_y - pred,2))/sum(pow(valid_y, 2))
        i_list.append(i)
        r2_list.append(score)
    
    max_index =  r2_list.index(max(r2_list))
    i = i_list[max_index]  
    trn = (year.iloc[0:i,:])['year'].to_list()
    train = df2[df2.year.isin(trn)]
    trn_x = train.drop(['RET','year'],axis=1)
    trn_y = train.RET
    
    tst = (year.iloc[i+12:i+13,:])['year'].to_list()
    test= df2[df2.year.isin(tst)]
    test_x = test.drop(['RET','year'],axis=1)
    test_y = test.RET.reset_index(drop=True)
    #best = HuberRegressor().fit(trn_x[['mvel1','bm','mom12m']] ,trn_y)
    regression = sm.RLM(trn_y,trn_x[['mvel1','bm','mom12m']],M=sm.robust.norms.HuberT() ) 
    best = regression.fit()
    ret_pred = pd.Series(best.predict(test_x[['mvel1','bm','mom12m']])).reset_index(drop=True)
    
    result = 1-sum(pow(test_y - ret_pred,2))/sum(pow(test_y, 2))
    return result

ols3_all = recursive_eval_OLS3(df2)
ols3_top = recursive_eval_OLS3(top1000)
ols3_bot = recursive_eval_OLS3(bot1000)
#store the result for model OLS3
total_output = total_output.append([{'Model': 'OLS_3+H'}],ignore_index=True)
total_output.loc[0,'All']=ols3_all*100
total_output.loc[0,'bot']=ols3_bot*100
total_output.loc[0,'top']=ols3_top*100

##PCR
#recursive performance evaluation scheme
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale 
import numpy as np
import matplotlib.pyplot as plt

def recursive_eval_PCR(df2):
    i_list = list()
    r2_list = list()
    importance = np.empty([20,])
    for i in range(18, 48):
        #seprate the data
        trn = (year.iloc[0:i,:])['year'].to_list()
        train = df2[df2.year.isin(trn)]
        trn_x = train.drop(['RET','year'],axis=1)
        trn_y = train.RET
        
        vd = (year.iloc[i:i+12,:])['year'].to_list()
        valid = df2[df2.year.isin(vd)]
        valid_x = valid.drop(['RET','year'],axis=1)
        valid_y = valid.RET.reset_index(drop=True) 
              
        pca2 = PCA()
        #pca2.explained_variance_ratio_
        X_reduced_train = pca2.fit_transform(scale(trn_x))[:,:20]
        # Train regression model on training data 
        regr = LinearRegression()
        regr.fit(X_reduced_train, trn_y)
        
        X_reduced_valid = pca2.fit_transform(scale(valid_x))[:,:20]
        pred = pd.Series(regr.predict(X_reduced_valid)).reset_index(drop=True)
        score = 1-sum(pow(valid_y - pred,2))/sum(pow(valid_y, 2))
        i_list.append(i)
        r2_list.append(score)
        # get importance and sum it up for each training loop
        # normalize the importance array
        varIMP = regr.coef_
        normal_VI = ( varIMP -  varIMP.min()) / ( varIMP -  varIMP.min()).sum()
        importance += normal_VI

    
    #calculate out the mean of variable importance
    importance = importance/len(range(18,48))
    importance = ( importance -  importance.min()) / ( importance -  importance.min()).sum()
    
    max_index =  r2_list.index(max(r2_list))
    i = i_list[max_index]  
    trn = (year.iloc[0:i,:])['year'].to_list()
    train = df2[df2.year.isin(trn)]
    trn_x = train.drop(['RET','year'],axis=1)
    trn_y = train.RET
    
    tst = (year.iloc[i+12:i+13,:])['year'].to_list()
    test= df2[df2.year.isin(tst)]
    test_x = test.drop(['RET','year'],axis=1)
    test_y = test.RET.reset_index(drop=True)
    
    pca = PCA()
    X_reduced_train1 = pca.fit_transform(scale(trn_x))[:,:20]
    regr1 = LinearRegression()
    regr1.fit(X_reduced_train1, trn_y)
    X_reduced_test = pca.fit_transform(scale(test_x))[:,:20]
    ret_pred = pd.Series(regr1.predict(X_reduced_test)).reset_index(drop=True)
    
    result = 1-sum(pow(test_y - ret_pred,2))/sum(pow(test_y, 2))
    importance = pd.DataFrame(importance,index = test_x.columns[0:20])
   
    return result,importance

PCR_all , PCR_importance = recursive_eval_PCR(df2)
#plot importance

PCR_importance.sort_values(0, axis=0, ascending=True).plot(kind='barh', color='b', )
plt.xlabel('PCR Variable Importance')
plt.gca().legend_ = None


PCR_top , _ = recursive_eval_PCR(top1000)
PCR_bot , _ = recursive_eval_PCR(bot1000)

total_output = total_output.append([{'Model': 'PCR'}],ignore_index=True)
total_output.loc[1,'All']=PCR_all*100
total_output.loc[1,'bot']=PCR_bot*100
total_output.loc[1,'top']=PCR_top*100

##PLS
#recursive performance evaluation scheme
from sklearn.cross_decomposition import PLSRegression

def recursive_eval_PLS(df2):
    i_list = list()
    r2_list = list()
    importance = np.empty([180,1])
    for i in range(18, 48):
        #seprate the data
        trn = (year.iloc[0:i,:])['year'].to_list()
        train = df2[df2.year.isin(trn)]
        trn_x = train.drop(['RET','year'],axis=1)
        trn_y = train.RET
        
        vd = (year.iloc[i:i+12,:])['year'].to_list()
        valid = df2[df2.year.isin(vd)]
        valid_x = valid.drop(['RET','year'],axis=1)
        valid_y = valid.RET.reset_index(drop=True) 
             
        # Train regression model on training data 
        pls = PLSRegression(n_components=3)
        pls.fit(trn_x, trn_y)
        pred = pd.DataFrame(pls.predict(valid_x)).iloc[:,0]
        score = 1-sum(pow(valid_y - pred,2))/sum(pow(valid_y, 2))
        i_list.append(i)
        r2_list.append(score)
        # get importance and sum it up for each training loop
        # normalize the importance array
        varIMP = pls.coef_
        normal_VI = ( varIMP -  varIMP.min()) / ( varIMP -  varIMP.min()).sum()
        importance += normal_VI
   
    #calculate out the mean of variable importance
    importance = importance/len(range(18,48))
    importance = ( importance -  importance.min()) / ( importance -  importance.min()).sum()
    
    max_index =  r2_list.index(max(r2_list))
    i = i_list[max_index]  
    trn = (year.iloc[0:i,:])['year'].to_list()
    train = df2[df2.year.isin(trn)]
    trn_x = train.drop(['RET','year'],axis=1)
    trn_y = train.RET
    
    tst = (year.iloc[i+12:i+13,:])['year'].to_list()
    test= df2[df2.year.isin(tst)]
    test_x = test.drop(['RET','year'],axis=1)
    test_y = test.RET.reset_index(drop=True)
    
    pls1 = PLSRegression(n_components=3)
    pls1.fit(trn_x, trn_y)
    ret_pred = pd.DataFrame(pls.predict(test_x)).iloc[:,0]
    
    result = 1-sum(pow(test_y - ret_pred,2))/sum(pow(test_y, 2))
    importance = pd.DataFrame(importance,index= test_x.columns)
    
    return result,importance

PLS_all , PLS_importance = recursive_eval_PLS(df2)
#plot importance
PLS_importance = PLS_importance.sort_values(0, axis=0, ascending=True)
plot = PLS_importance.iloc[160:180,:]
plot.plot(kind='barh', color='b')
plt.xlabel('PLS Variable Importance')
plt.gca().legend_ = None

PLS_top , _ = recursive_eval_PLS(top1000)
PLS_bot , _ = recursive_eval_PLS(bot1000)

total_output = total_output.append([{'Model': 'PLS'}],ignore_index=True)
total_output.loc[2,'All']=PLS_all*100
total_output.loc[2,'bot']=PLS_bot*100
total_output.loc[2,'top']=PLS_top*100

##Elastic net
#recursive performance evaluation scheme
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
def recursive_eval_elasticnet(df2):
    i_list = list()
    r2_list = list()
    for i in range(18, 48):
        # seprate the data
        trn = (year.iloc[0:i, :])['year'].to_list()
        train = df2[df2.year.isin(trn)]
        trn_x = train.drop(['RET'], axis=1)
        trn_y = train.RET

        vd = (year.iloc[i:i + 12, :])['year'].to_list()
        valid = df2[df2.year.isin(vd)]
        valid_x = valid.drop(['RET'], axis=1)
        valid_y = valid.RET.reset_index(drop=True)

        grid = 10 ** (-4)

        regr = ElasticNet(alpha=grid, l1_ratio=0.5, max_iter=3000)
        regr.fit(trn_x, trn_y)
        pred = pd.Series(regr.predict(valid_x)).reset_index(drop=True)
        score = 1 - sum(pow(valid_y - pred, 2)) / sum(pow(valid_y, 2))
        i_list.append(i)
        r2_list.append(score)

    max_index = r2_list.index(max(r2_list))
    j = i_list[max_index]

    tst = (year.iloc[j + 12:j + 13, :])['year'].to_list()
    test = df2[df2.year.isin(tst)]
    test_x = test.drop(['RET'], axis=1)
    test_y = test.RET.reset_index(drop=True)

    grid = 10 ** (-4)

    ret_regr = ElasticNet(alpha=grid, l1_ratio=0.5, max_iter=3000)
    ret_regr.fit(test_x, test_y)
    ret_pred = pd.Series(ret_regr.predict(test_x)).reset_index(drop=True)

    result = 1 - sum(pow(test_y - ret_pred, 2)) / sum(pow(test_y, 2))
    importance = ret_regr.coef_
    
    return result,importance


EN_all, EN_importance = recursive_eval_elasticnet(df2)
EN_importance = pd.DataFrame(EN_importance, index=df2.drop(['RET'],axis=1).columns)
EN_importance = EN_importance.sort_values(0, axis=0, ascending=True)
pt = EN_importance.iloc[160:180,:]
pt.plot(kind='barh', color='b', )
plt.xlabel('ElasticNet Variable Importance')
plt.gca().legend_ = None
    
EN_top, _ = recursive_eval_elasticnet(top1000)
EN_bot, _ = recursive_eval_elasticnet(bot1000)

total_output = total_output.append([{'Model': 'ENet'}],ignore_index=True)
total_output.loc[3,'All']=EN_all*100
total_output.loc[3,'bot']=EN_bot*100
total_output.loc[3,'top']=EN_top*100

##Random Forest
#recursive performance evaluation scheme
from sklearn.ensemble import RandomForestRegressor
def recursive_eval_rf(df2):
    i_list = list()
    r2_list = list()
    for i in range(18, 48):
        # seprate the data
        trn = (year.iloc[0:i, :])['year'].to_list()
        train = df2[df2.year.isin(trn)]
        trn_x = train.drop(['RET'], axis=1)
        trn_y = train.RET

        vd = (year.iloc[i:i + 12, :])['year'].to_list()
        valid = df2[df2.year.isin(vd)]
        valid_x = valid.drop(['RET'], axis=1)
        valid_y = valid.RET.reset_index(drop=True)

        regr = RandomForestRegressor(n_estimators=300,  max_depth=6, max_features=3)
        regr.fit(trn_x, trn_y)
        pred = pd.Series(regr.predict(valid_x)).reset_index(drop=True)
        score = 1 - sum(pow(valid_y - pred, 2)) / sum(pow(valid_y, 2))
        i_list.append(i)
        r2_list.append(score)

    max_index = r2_list.index(max(r2_list))
    j = i_list[max_index]

    tst = (year.iloc[j + 12:j + 13, :])['year'].to_list()
    test = df2[df2.year.isin(tst)]
    test_x = test.drop(['RET'], axis=1)
    test_y = test.RET.reset_index(drop=True)

    ret_regr = RandomForestRegressor(n_estimators=300,  max_depth=6, max_features=3)
    ret_regr.fit(test_x, test_y)
    ret_pred = pd.Series(ret_regr.predict(test_x)).reset_index(drop=True)

    result = 1 - sum(pow(test_y - ret_pred, 2)) / sum(pow(test_y, 2))

    importance = pd.DataFrame({'Importance': ret_regr.feature_importances_* 100}, index=test_x.columns)
    importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='b', )
    plt.xlabel('Variable Importance')
    plt.gca().legend_ = None

    return result


RF_all = recursive_eval_rf(df2)
RF_top = recursive_eval_rf(top1000)
RF_bot = recursive_eval_rf(bot1000)

total_output = total_output.append([{'Model': 'RF'}],ignore_index=True)
total_output.loc[4,'All']=RF_all *100
total_output.loc[4,'bot']=RF_bot*100
total_output.loc[4,'top']=RF_top*100

##GBRT
#recursive performance evaluation scheme
from sklearn.ensemble import GradientBoostingRegressor

def recursive_eval_GBRT(df2):
    i_list = list()
    r2_list = list()
    for i in range(18, 48):
        #seprate the data
        trn = (year.iloc[0:i,:])['year'].to_list()
        train = df2[df2.year.isin(trn)]
        trn_x = train.drop(['RET','year'],axis=1)
        trn_y = train.RET
        
        vd = (year.iloc[i:i+12,:])['year'].to_list()
        valid = df2[df2.year.isin(vd)]
        valid_x = valid.drop(['RET','year'],axis=1)
        valid_y = valid.RET.reset_index(drop=True) 
             
        # Train regression model on training data 
        model = GradientBoostingRegressor(n_estimators = 1000,loss='huber',learning_rate = 0.1,max_depth = 2)
        model.fit(trn_x, trn_y)
        pred = pd.Series(model.predict(valid_x)).reset_index(drop=True)
        score = 1-sum(pow(valid_y - pred,2))/sum(pow(valid_y, 2))
        i_list.append(i)
        r2_list.append(score)
   

    max_index =  r2_list.index(max(r2_list))
    i = i_list[max_index]  
    trn = (year.iloc[0:i,:])['year'].to_list()
    train = df2[df2.year.isin(trn)]
    trn_x = train.drop(['RET','year'],axis=1)
    trn_y = train.RET
    
    tst = (year.iloc[i+12:i+13,:])['year'].to_list()
    test= df2[df2.year.isin(tst)]
    test_x = test.drop(['RET','year'],axis=1)
    test_y = test.RET.reset_index(drop=True)
    
    model1 = GradientBoostingRegressor(n_estimators = 1000,loss='huber',learning_rate = 0.1,max_depth = 2)
    model1.fit(trn_x, trn_y)
    ret_pred = pd.Series(model1.predict(test_x)).reset_index(drop=True)
    
    result = 1-sum(pow(test_y - ret_pred,2))/sum(pow(test_y, 2))
    importance = model1.feature_importances_
    
    return result,importance

GBRT_all , GBRT_importance = recursive_eval_GBRT(df2)
#plot importance
GBRT_importance = pd.DataFrame(GBRT_importance,index= df2.drop(['RET','year'],axis=1).columns)
GBRT_importance = GBRT_importance.sort_values(0, axis=0, ascending=True)
plot = GBRT_importance.iloc[160:180,:]
plot.plot(kind='barh', color='b')
plt.xlabel('GBRT Variable Importance')
plt.gca().legend_ = None

GBRT_top , _ = recursive_eval_GBRT(top1000)
GBRT_bot , _ = recursive_eval_GBRT(bot1000)

total_output = total_output.append([{'Model': 'GBRT+H'}],ignore_index=True)
total_output.loc[5,'All']=GBRT_all*100
total_output.loc[5,'bot']=GBRT_bot*100
total_output.loc[5,'top']=GBRT_top*100

#plot out the total output
import matplotlib.pyplot as plt
size = 6
 
x = np.arange(size)

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2
 
plt.bar(x, total_output['All'],  width=width, label='All')
plt.bar(x + width, total_output['top'], width=width, label='Top')
plt.bar(x + 2 * width, total_output['bot'], width=width, label='Bot')
plt.xticks(x, labels=total_output['Model'])
plt.legend()
plt.show()
