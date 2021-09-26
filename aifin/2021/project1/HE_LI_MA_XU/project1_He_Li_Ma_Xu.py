# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 20:29:33 2021

@author: MJK
"""

import pandas as pd
import numpy as np
from pandas import DataFrame
import os

############### Buearu Data Clean ###############
pd.set_option('display.max_columns', None)
"""
CREDIT_DAY_OVERDUE:Number of days past due on CB credit at the time of application for related loan in our sample
AMT_CREDIT_SUM:Current credit amount for the Credit Bureau credit
AMT_CREDIT_SUM_OVERDUE:Current amount overdue on Credit Bureau credit
DAYS_CREDIT_UPDATE:How many days before loan application did last information about the Credit Bureau credit come
"""
data = pd.read_csv("./bureau.csv")[["SK_ID_CURR","CREDIT_ACTIVE","CREDIT_DAY_OVERDUE",
                                   "AMT_CREDIT_SUM","AMT_CREDIT_SUM_OVERDUE","DAYS_CREDIT_UPDATE"]]
count = pd.read_excel("./count.xlsx")
print(data.describe())

data = data.sort_values("DAYS_CREDIT_UPDATE",ascending=False)
group = data.groupby(["SK_ID_CURR"]).head(1).sort_values("SK_ID_CURR",ascending=True).drop("DAYS_CREDIT_UPDATE",axis=1)
group["CREDIT_ACTIVE"] = pd.Categorical(group["CREDIT_ACTIVE"]).codes
# FILL in NA value; for CREDIT_ACTIVE use NA; for others use 25% quantile
buearu_summ = pd.concat([count.set_index("SK_ID_CURR"),group.set_index("SK_ID_CURR")],axis=1, join='outer')
buearu_summ["CREDIT_ACTIVE"].fillna(-1,inplace=True)
buearu_summ["CREDIT_DAY_OVERDUE"].fillna(buearu_summ["CREDIT_DAY_OVERDUE"].quantile(0.25),inplace=True)
buearu_summ["AMT_CREDIT_SUM"].fillna(buearu_summ["AMT_CREDIT_SUM"].quantile(0.25),inplace=True)
buearu_summ["AMT_CREDIT_SUM_OVERDUE"].fillna(buearu_summ["AMT_CREDIT_SUM_OVERDUE"].quantile(0.25),inplace=True)

buearu_summ.to_excel("buearu features.xlsx")
dir_path = 'D:\新建文件夹'  


fnlists = os.listdir(dir_path)
df = pd.DataFrame()

for fn in fnlists:
    if 'csv' in fn:
        path = os.path.join(dir_path, fn)
        df_tem = pd.read_csv(path)
        gb = df_tem['SK_ID_CURR'].value_counts().reset_index()
        gb.columns = ['SK_ID_CURR','count']
        df = pd.concat([df,gb])
res = df.groupby('SK_ID_CURR')['count'].sum().reset_index()


res = res.set_index('SK_ID_CURR')
res = res.reindex(list(range(100001,456256)))
res['count'] = res['count'].fillna(0)
res = res.reset_index()
print(res)

res.to_excel(os.path.join(dir_path,'output.xlsx'),index=False)






###################### Application data clean
def clean(data1: DataFrame, data2: DataFrame):
    if 'TARGET' in data1.columns:
        col_f_train = ['SK_ID_CURR', 'TARGET','CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 
                 'NAME_EDUCATION_TYPE', 'DAYS_EMPLOYED', 'FLAG_MOBIL', 
                 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE',
                 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
                 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 
                 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', 
                 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
                 'DAYS_LAST_PHONE_CHANGE']
        data_new = data1[col_f_train]
        for i in range(2, 22):
            num = str(i)
            data_new = pd.concat([data_new, data1['FLAG_DOCUMENT_{}'.format(num)]], axis = 1)
        nan_num = []
        for cols in data_new.columns :
            nan_num.append(data_new[cols].isna().sum())
        nan_num = pd.concat([pd.DataFrame(data_new.columns), pd.DataFrame(nan_num)], axis = 1)
        data_new[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']] = data_new[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].fillna(0)
        data_new = data_new.drop(['OCCUPATION_TYPE'], axis = 1)
        data_new = data_new.drop(['ORGANIZATION_TYPE'], axis = 1)
        data_new[['OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']] = data_new[['OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']].fillna(0)
        data_new['CODE_GENDER'] = [0 if data_new['CODE_GENDER'].iloc[i] == 'F' else 1 for i in range(len(data_new))]
        data_new['FLAG_OWN_CAR'] = [0 if data_new['FLAG_OWN_CAR'].iloc[i] == 'N' else 1 for i in range(len(data_new))]
        data_new['FLAG_OWN_REALTY'] = [0 if data_new['FLAG_OWN_REALTY'].iloc[i] == 'N' else 1 for i in range(len(data_new))]
        if len(np.unique(data_new['NAME_INCOME_TYPE'])) == 8:
            data_new['NAME_INCOME_TYPE'] = data_new['NAME_INCOME_TYPE'].replace(list(np.unique(data_new['NAME_INCOME_TYPE'])), [6, 7, 2, 3, 5, 4, 1, 0])
        else:
            data_new['NAME_INCOME_TYPE'] = data_new['NAME_INCOME_TYPE'].replace(list(np.unique(data_new['NAME_INCOME_TYPE'])), [5, 6, 2, 4, 1, 0, 3])
        data_new['NAME_EDUCATION_TYPE'] = data_new['NAME_EDUCATION_TYPE'].replace(list(np.unique(data_new['NAME_EDUCATION_TYPE'])), [4, 3, 2, 0, 1])
        data_new['AVG_REGION_RATING_CLIENT'] = (data_new['REGION_RATING_CLIENT'] + data_new['REGION_RATING_CLIENT_W_CITY'])/2
        data_new['AVG_FLAG_PHONE'] = data_new.iloc[:,9:14].mean(axis=1)
        data_new['SUM_LIVE_CITY_WORK_CITY'] = data_new.iloc[:,17:23].sum(axis=1)
        data_new['SUM_EXT_SOURCE'] = data_new.iloc[:,23:26].sum(axis=1)
        data_new['SUM_OBS_DEF'] = data_new.iloc[:,26:28].sum(axis=1)
        data_new['SUM_FLAG_DOCUMENT'] = data_new.iloc[:,28:49].sum(axis=1)
        col1 = data_new['CNT_FAM_MEMBERS']
        col2 = data_new['DAYS_LAST_PHONE_CHANGE']
        data_new.drop(data_new.columns[8:49], axis=1, inplace=True)
        data_new = pd.concat([data_new,col1,col2], axis=1)
        data2_new = data2[data2['SK_ID_CURR'].isin(data_new['SK_ID_CURR'])].reset_index(drop = True)
        data_new = pd.concat([data_new, data2_new.iloc[:, 1:]], axis = 1)
        data_new = data_new.dropna(axis = 0, how = 'any').reset_index(drop = True)
        
    else:
        col_f_train = ['SK_ID_CURR','CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 
                 'NAME_EDUCATION_TYPE', 'DAYS_EMPLOYED', 'FLAG_MOBIL', 
                 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE',
                 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
                 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 
                 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', 
                 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
                 'DAYS_LAST_PHONE_CHANGE']
        data_new = data1[col_f_train]
        for i in range(2, 22):
            num = str(i)
            data_new = pd.concat([data_new, data1['FLAG_DOCUMENT_{}'.format(num)]], axis = 1)
        nan_num = []
        for cols in data_new.columns :
            nan_num.append(data_new[cols].isna().sum())
        nan_num = pd.concat([pd.DataFrame(data_new.columns), pd.DataFrame(nan_num)], axis = 1)
        data_new[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']] = data_new[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].fillna(0)
        data_new = data_new.drop(['OCCUPATION_TYPE'], axis = 1)
        data_new = data_new.drop(['ORGANIZATION_TYPE'], axis = 1)
        data_new[['OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']] = data_new[['OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']].fillna(0)
        data_new['CODE_GENDER'] = [0 if data_new['CODE_GENDER'].iloc[i] == 'F' else 1 for i in range(len(data_new))]
        data_new['FLAG_OWN_CAR'] = [0 if data_new['FLAG_OWN_CAR'].iloc[i] == 'N' else 1 for i in range(len(data_new))]
        data_new['FLAG_OWN_REALTY'] = [0 if data_new['FLAG_OWN_REALTY'].iloc[i] == 'N' else 1 for i in range(len(data_new))]
        if len(np.unique(data_new['NAME_INCOME_TYPE'])) == 8:
            data_new['NAME_INCOME_TYPE'] = data_new['NAME_INCOME_TYPE'].replace(list(np.unique(data_new['NAME_INCOME_TYPE'])), [6, 7, 2, 3, 5, 4, 1, 0])
        else:
            data_new['NAME_INCOME_TYPE'] = data_new['NAME_INCOME_TYPE'].replace(list(np.unique(data_new['NAME_INCOME_TYPE'])), [5, 6, 2, 4, 1, 0, 3])
        data_new['NAME_EDUCATION_TYPE'] = data_new['NAME_EDUCATION_TYPE'].replace(list(np.unique(data_new['NAME_EDUCATION_TYPE'])), [4, 3, 2, 0, 1])
        data_new['AVG_REGION_RATING_CLIENT'] = (data_new['REGION_RATING_CLIENT'] + data_new['REGION_RATING_CLIENT_W_CITY'])/2
        data_new['AVG_FLAG_PHONE'] = data_new.iloc[:,7:12].mean(axis=1)
        data_new['SUM_LIVE_CITY_WORK_CITY'] = data_new.iloc[:,15:21].sum(axis=1)
        data_new['SUM_EXT_SOURCE'] = data_new.iloc[:,21:24].sum(axis=1)
        data_new['SUM_OBS_DEF'] = data_new.iloc[:,24:26].sum(axis=1)
        data_new['SUM_FLAG_DOCUMENT'] = data_new.iloc[:,26:47].sum(axis=1)
        col1 = data_new['CNT_FAM_MEMBERS']
        col2 = data_new['DAYS_LAST_PHONE_CHANGE']
        data_new.drop(data_new.columns[6:47], axis=1, inplace=True)
        data_new = pd.concat([data_new,col1,col2], axis=1)
        data2_new = data2[data2['SK_ID_CURR'].isin(data_new['SK_ID_CURR'])].reset_index(drop = True)
        data_new = pd.concat([data_new, data2_new.iloc[:, 1:]], axis = 1)
        
        
    return data_new



if __name__ == '__main__':
    
    
    
    
    
    
    
    
    
    ############# Logistic Regression ##########################
    data1 = pd.read_csv(r'D:\googledownloads\HKUST\6010Z\projects\application_train.csv')
    data2 = pd.read_excel(r'D:\googledownloads\HKUST\6010Z\projects\buearu features.xlsx')
    data_clean = clean(data1, data2)
    nan_num = []
    for cols in data_clean.columns :
        nan_num.append(data_clean[cols].isna().sum())
    nan_num = pd.concat([pd.DataFrame(data_clean.columns), pd.DataFrame(nan_num)], axis = 1)
        
    X = data_clean.iloc[:,2:]
    y = data_clean.iloc[:,1]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    
    from sklearn.preprocessing import StandardScaler
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=10,penalty='l2')
    lr.fit(X_train_std, y_train)
    print('Training accuracy:', lr.score(X_train_std, y_train))
    print('Test accuracy:', lr.score(X_test_std, y_test))
    
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    # 计算 预测率
    probas = lr.fit(X_train, y_train).predict_proba(X_test)
    # 计算 fpr,tpr    
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr,  tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    print('ROC:', roc_auc)
    
    from sklearn.model_selection import cross_val_score
    print(cross_val_score(lr, X_train, y_train, cv=5, scoring="roc_auc"))
    
    data3 = pd.read_csv(r'D:\googledownloads\HKUST\6010Z\projects\application_test.csv')
    data_test_clean = clean(data3,data2)
    
    y_pre_test = lr.predict(data_test_clean.iloc[:,1:])
    output = pd.DataFrame(columns = ['SK_ID_CURR','predict'])
    output['SK_ID_CURR'] = data_test_clean['SK_ID_CURR']       
    output['predict'] = y_pre_test

##################### Knn #####################################
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    data1 = pd.read_csv(r'D:\codes\application_train.csv')
    data2 = pd.read_excel(r'D:\codes\buearu features.xlsx')
    
    data_clean = clean(data1, data2)

    train_x = data_clean.copy(deep=True)
    del train_x['TARGET']
    del train_x['SK_ID_CURR']
    del train_x['AVG_FLAG_PHONE']
    del train_x['SUM_OBS_DEF']
    train_y = data_clean['TARGET']
  
    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.3)
    
    standardScaler = StandardScaler()  
    standardScaler.fit(x_train) 
    x_train = standardScaler.transform(x_train) 
    x_test_standard = standardScaler.transform(x_test)
    
    # 计算k从5到10的准确度得分  
    score_list=pd.DataFrame(np.random.rand(6).reshape((6,1)),columns=['score'],index=['5','6','7','8','9','10'],dtype='double')
    for i in range(5,11):
        
       knn = KNeighborsClassifier(n_neighbors = i)    #default_n_neighbors=5  
       knn.fit(x_train, y_train) 
       y_predict = knn.predict(x_test_standard)
       score = accuracy_score(y_test,y_predict)
       print("k-value:",i,"score is",score)
       score_list['score'][i-5]=score
       
################# Random Forest ###############
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import pandas as pd
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import pandas as pd
    
    train = pd.read_excel(".//train.xlsx").set_index("SK_ID_CURR")
    test = pd.read_excel(".//test.xlsx").set_index("SK_ID_CURR")
    train = train.drop(["AVG_FLAG_PHONE","SUM_OBS_DEF"],axis=1)
    y = train["TARGET"]
    X = train.drop("TARGET",axis=1)
    test = test.drop(["AVG_FLAG_PHONE","SUM_OBS_DEF"],axis=1)
    
    """
    Parameter tuning
    for n in range(205,306,10):
        print("n_estimators",n)  
        model = RandomForestClassifier(n_estimators=205, max_depth=11,criterion="entropy",class_weight="balanced",min_samples_split=35)
    
    # model.fit(X,y)
    # predict_test = mode1,sum((predict_test- data["TARGET"][200000:])==0)/len(predict_test - data["TARGET"][200000:]))
        scores1 = cross_val_score(model, data.drop("TARGET",axis=1), data["TARGET"],cv=6,scoring="roc_auc")
        print("交叉验证",scores1.mean())
    """
        
    model = RandomForestClassifier(n_estimators=105, max_depth=11,
                                   criterion="entropy",class_weight="balanced",
                                   min_samples_split=35)
    
    
    model.fit(X,y)
    result = model.predict(test)
    classification = pd.DataFrame(result,index=test.index,columns=["TARGET"])
    classification.to_csv("result.csv")

######################## Gradient Boosting ###########################
    data1 = pd.read_csv(r'C:\Users\Administrator\Desktop\home credit\application_train.csv')
    data2 = pd.read_csv(r'C:\Users\Administrator\Desktop\home credit\buearu features.csv')
    data_clean = clean(data1, data2)
    
    from sklearn import metrics
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    x = data_clean.iloc[:, 2:]
    y = data_clean['TARGET'].ravel()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
    gdbt = GradientBoostingClassifier(learning_rate= 0.1, min_samples_split= 5000, min_samples_leaf= 5,
                                      max_depth= 3  , loss= 'deviance')
    gdbt.fit(x_train, y_train.ravel())
    y_pre = gdbt.predict(x_test)
    y_preprob = gdbt.predict_proba(x_test)[:, 1]
    score = cross_val_score(gdbt, x, y, scoring = 'roc_auc')
    
    data3 = pd.read_csv(r'C:\Users\Administrator\Desktop\home credit\application_test.csv')
    data_test_clean = clean(data3, data2)
    y_pre_test = gdbt.predict(data_test_clean.iloc[:,1:])
    

