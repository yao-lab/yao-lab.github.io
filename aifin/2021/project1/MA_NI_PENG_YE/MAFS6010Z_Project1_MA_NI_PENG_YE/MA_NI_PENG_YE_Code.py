# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


df_train = pd.read_csv('application_train.csv')
df_test = pd.read_csv('application_test.csv')
#when we exploring or cleaning the data we need to do it on both trian and test case
#drop all features that have >60% missing data
df_train=df_train.dropna(thresh=len(df_train)*0.4, axis=1)


#create a new feature states the completeness of application file for each cilent
df_train['incomplete'] = 1
df_train.loc[df_train.isnull().sum(axis=1) < 40, 'incomplete'] = 0
df_test['incomplete'] = 1
df_test.loc[df_test.isnull().sum(axis=1) < 40, 'incomplete'] = 0


#deal with unknown data in gender feature
print(list(df_train['CODE_GENDER'].unique()))
df_train['CODE_GENDER'].replace({'XNA': np.nan}, inplace = True)
df_test['CODE_GENDER'].replace({'XNA': np.nan}, inplace = True)


#tramsform daily date to years, and
df_train['YEARS_EMPLOYED'] = df_train['DAYS_EMPLOYED'].apply(lambda x: int(x/-365))
df_test['YEARS_EMPLOYED'] = df_test['DAYS_EMPLOYED'].apply(lambda x: int(x/-365))
df_train['YEARS_EMPLOYED'].replace({-1000: np.nan}, inplace = True)
df_test['YEARS_EMPLOYED'].replace({-1000: np.nan}, inplace = True)

#Replace abnormal employ date with NaN，and label a new feature state abnormal issue
df_train['DAYS_EMPLOYED_ANOM'] = df_train["DAYS_EMPLOYED"] == 365243
df_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

df_test['DAYS_EMPLOYED_ANOM'] = df_test["DAYS_EMPLOYED"] == 365243
df_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)


#drop daily date we dont need
df_train = df_train.drop(['DAYS_EMPLOYED'], axis=1)
df_test = df_test.drop(['DAYS_EMPLOYED'], axis=1)
#transform ojective type feature to numerical feature by using labelencoder in sklearn package
#objective type feature encoding，and store/count those objective feature
from sklearn import preprocessing
label = preprocessing.LabelEncoder()
label_count = 0
label_list = []
#check out objective feature that cant use pd.get_dummies
for col in df_test:
    if df_test[col].dtype == 'object' or df_test[col].dtype == 'bool':
        try:
            label.fit(df_train[col])
            # Transform both training and testing data
            df_train[col] = label.transform(df_train[col])
            df_test[col] = label.transform(df_test[col])
        except:#if we cant encode the objective, we need to drop it
            print('error exist')# print the error messages out
            if col in df_train.columns.values.tolist():
                df_train = df_train.drop([col], axis=1)
            if col in df_test.columns.values.tolist():
                df_test = df_test.drop([col], axis=1)
        else:
            # counting number and storing features
            label_count += 1
            label_list.append(col)
        
#print the num of features we transform
print('%d columns were label encoded.' % label_count)   

#create some self-design features that is meaningful
#term : credit/annuity
df_train['TERM'] = df_train['AMT_CREDIT'] / df_train['AMT_ANNUITY']
df_test['TERM'] = df_test['AMT_CREDIT'] / df_test['AMT_ANNUITY']
#percentage of credit by income
df_train['CREDIT_INCOME_PERCENT'] = df_train['AMT_CREDIT'] / df_train['AMT_INCOME_TOTAL']
#percentage of annuity by income
df_train['ANNUITY_INCOME_PERCENT'] = df_train['AMT_ANNUITY'] / df_train['AMT_INCOME_TOTAL']

#transform days_birth/days_employed to years type and take its lower boundary
df_train['YEARS_BIRTH'] = df_train['DAYS_BIRTH'].apply(lambda x: int(x/-365))
df_test['YEARS_BIRTH'] = df_test['DAYS_BIRTH'].apply(lambda x: int(x/-365))
df_train = df_train.drop(['DAYS_BIRTH'], axis=1)
df_test = df_test.drop(['DAYS_BIRTH'], axis=1)

#percentage of employed day by age
df_train['YEARS_EMPLOYED_PERCENT'] = df_train['YEARS_EMPLOYED'] / df_train['YEARS_BIRTH']


#input data and rename the colname into legal type
import re
df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df_test = df_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#checking missing data and print it out
def find_missing(data):
    # number of missing values
    count_missing = data.isnull().sum().values
    # total records
    total = data.shape[0]
    # percentage of missing
    ratio_missing = count_missing/total
    # return a dataframe to show: feature name, # of missing and % of missing
    return pd.DataFrame(data={'missing_count':count_missing, 'missing_ratio':ratio_missing}, index=data.columns.values)
#sort the missing_percent descending
missing_percent=find_missing(df_train).sort_values(by=['missing_ratio'],ascending=False)
print(missing_percent.head(12))
#impute missing part by fillna method with strategy median
def refill(data):
    for col in data.columns.values.tolist():
        data[col] = data[col].fillna(data[col].median())
    return data
df_train = refill(df_train)
df_test = refill(df_test)
#set up data for modeling,and drop prediction col and index col
X = df_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y = df_train.TARGET
predition_X = df_test.drop(['SK_ID_CURR'], axis=1)


#using 5-folds cv to improve the modeling
#using package to randomly split data into 5 piece
folds = StratifiedKFold(n_splits=5,shuffle=True,random_state=6)
preds = np.zeros(X.shape[0])
predictions = np.zeros(predition_X.shape[0])

#perform the cv
valid_score = 0
for n_fold, (train_idx, eval_idx) in enumerate(folds.split(X, y)):
    train_x, train_y = X.iloc[train_idx], y[train_idx]
    eval_x, eval_y = X.iloc[eval_idx], y[eval_idx]    
    
    train_data = lgb.Dataset(data=train_x, label=train_y,categorical_feature=label_list)
    eval_data = lgb.Dataset(data=eval_x, label=eval_y)
    
    params = {'application':'binary','num_iterations':4000, 'learning_rate':0.05, 'num_leaves':24, 
             'feature_fraction':0.8, 'bagging_fraction':0.9,
             'lambda_l1':0.1, 'lambda_l2':0.1, 'min_split_gain':0.01, 'min_data_in_leaf':20,
             'early_stopping_round':100, 'max_depth':7, 
             'min_child_weight':40, 'metric':'auc','verbose' : -1}


    #train the model
    lgb_es_model = lgb.train(params, train_data, valid_sets=[train_data, eval_data], categorical_feature=label_list,verbose_eval=100) 
    #pick the best one
    preds[eval_idx] = lgb_es_model.predict(eval_x, num_iteration=lgb_es_model.best_iteration)
    #store the best prediction result 
    predictions += lgb_es_model.predict(predition_X, num_iteration=lgb_es_model.best_iteration,predict_disable_shape_check=True) / folds.n_splits
    #print out aue score to do the model assessment
    print('AUC of Fold %2d  : %.3f' % (n_fold + 1, roc_auc_score(eval_y, preds[eval_idx])))
    valid_score += roc_auc_score(eval_y, preds[eval_idx])

#calculate out the mean of cv score
print('valid score:', str(round(valid_score/folds.n_splits,4)))
#plot the feature importance graph to see which features is most important
lgb.plot_importance(lgb_es_model, height=0.5, max_num_features=20, ignore_zero = False, figsize = (12,6), importance_type ='gain')
#output the predictions as csv file
output = pd.DataFrame({'SK_ID_CURR':df_test.SK_ID_CURR, 'TARGET': predictions})
output.to_csv('submission.csv', index=False)
































