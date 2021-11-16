#This part is for fitting and testing the PCR, PLS, GBT and RF model, based on the data constructed in part 1

import pandas as pd
import numpy as np
import math
import lightgbm as lgb
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gc
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid,PredefinedSplit,GridSearchCV,RandomizedSearchCV
import scipy.stats.distributions as dists
import matplotlib.pyplot as plt
import pickle

#import data
df = pd.read_csv("df_merge_fill.csv", sep=",")
col_name = df.columns
gc.collect()

'''
PCR
'''
pcr_R_sq_list = []
pcr_feat_imp_package = pd.DataFrame()
pcr_feat_imp = pd.DataFrame()
for train_year in range(1974,2004):
    validation_year = train_year + 12 
    test_year = validation_year + 1
    non_predictor = ['yyyymm','RET','SHROUT','mve0','prc','permno','excess_return']

    # Hyper-parameter tuning (for every 5 years)
    if(train_year+1)%5 == 0:
        X = df[(df['yyyymm'] // 100 <= validation_year)].drop(columns=non_predictor)
        X = X.fillna(0)
        Y = df[(df['yyyymm'] // 100 <= validation_year)]['excess_return'].to_frame()
        number_of_train = sum(1 for yr in df['yyyymm'] if yr//100 <= train_year)
        split_index = np.repeat(0, len(X))
        split_index[0:number_of_train-1]=-1
        pds = PredefinedSplit(test_fold = split_index)
        pcr_param_grid = {'pca__n_components':[5,15,30,45]}
        pca_hpt = PCA()
        pcr_hpt = make_pipeline(StandardScaler(), pca_hpt, LinearRegression())
        pcr_hpt_f = GridSearchCV(estimator=pcr_hpt, cv=pds, param_grid=pcr_param_grid)
        pcr_hpt_fit = pcr_hpt_f.fit(X, Y)
        gc.collect()
        
    #Fit model
    X = df[(df['yyyymm'] // 100 <= train_year)].drop(columns=non_predictor)
    X = X.fillna(0)
    Y = df[(df['yyyymm'] // 100 <= train_year)]['excess_return'].to_frame()
    
    X_test = df[(df['yyyymm'] // 100 == test_year)].drop(columns=non_predictor)
    X_test = X_test.fillna(0)
    Y_test = df[(df['yyyymm'] // 100 == test_year)]['excess_return']
        
    n_comp = pcr_hpt_f.best_params_['pca_n_components']
    pca = PCA(n_components=n_comp)
    pcr = make_pipeline(StandardScaler(), pca, LinearRegression())
    pcr_fit = pcr.fit(X, Y)
        
    #Export model
    filename = "pcr_%s.pkl" % train_year
    with open(filename, 'wb') as fp:
        pickle.dump(pcr, fp)

    #Prediction and R2
    pcr_prediction = pcr_fit.predict(X_test).reshape(len(Y_test),1)
    error_pred = pcr_prediction-Y_test.values.reshape(len(Y_test),1)
    pcr_R2 = 1-sum(error_pred*error_pred)/sum(Y_test.to_numpy()*Y_test.to_numpy())
    pcr_R_sq_list = np.append(pcr_R_sq_list,np.array(pcr_R2))
       
    #Feature Importance
    # Extract feature names
    pcr_feature_names = list(X.columns)
    # First calculate full model R_sq
    y_hat = pcr_fit.predict(X).reshape(len(Y),1)
    error = y_hat-Y.values
    R_sq = 1-sum(error*error)/sum(Y.to_numpy()*Y.to_numpy())
    # Create a empty data for storing the feature importance
    pcr_R_sq_loss_table = X.loc[0,pcr_feature_names]
    pcr_R_sq_loss_table[pcr_feature_names] = 0  
        
    for pred in pcr_feature_names:
        # We adjust the y_hat by removing the term corresponding feature x features weight
        pred_temp = X[pred]
        X[pred] = 0
        y_hat_new = pcr_fit.predict(X).reshape(len(Y),1)
        X[pred] = pred_temp
        error_new = y_hat_new-Y.values.reshape(len(Y),1)
        R_sq_new = 1-sum(error_new*error_new)/sum(Y.to_numpy()*Y.to_numpy())
        #print(R_sq_new)
        # Compute loss in R_sq
        loss_in_R2 = R_sq - R_sq_new
        pcr_R_sq_loss_table[pred] = loss_in_R2
        
    pcr_feat_imp_temp = pcr_R_sq_loss_table/sum(pcr_R_sq_loss_table)
    pcr_feat_imp[test_year] = pcr_feat_imp_temp
    '''
    if test_year==1997:
        pcr_feat_imp.to_csv("pcr_feat_imp_1997.csv",index=True)
        pd.DataFrame(pcr_R_sq_list).to_csv("pcr_R_sq_list_1997.csv",index=False)
    if test_year==2007:
        pcr_feat_imp.to_csv("pcr_feat_imp_2007.csv",index=True)
        pd.DataFrame(pcr_R_sq_list).to_csv("pcr_R_sq_list_2007.csv",index=False)
    '''
    gc.collect()
pcr_feat_imp.to_csv("pcr_feat_imp.csv",index=True)
pd.DataFrame(pcr_R_sq_list).to_csv("pcr_R_sq_list.csv",index=False)


'''
PLS
'''
pls_R_sq_list = []
pls_feat_imp_package = pd.DataFrame()
pls_feat_imp = pd.DataFrame()
for train_year in range(1974,2004):
    validation_year = train_year + 12 
    test_year = validation_year + 1
    non_predictor = ['yyyymm','RET','SHROUT','mve0','prc','permno','excess_return']

    # Hyper-parameter tuning (for every 5 years)
    if(train_year+1)%5 == 0:
        X = df[(df['yyyymm'] // 100 <= validation_year)].drop(columns=non_predictor)
        X = X.fillna(0)
        Y = df[(df['yyyymm'] // 100 <= validation_year)]['excess_return'].to_frame()
        number_of_train = sum(1 for yr in df['yyyymm'] if yr//100 <= train_year)
        split_index = np.repeat(0, len(X))
        split_index[0:number_of_train-1]=-1
        pds = PredefinedSplit(test_fold = split_index)
        pls_param_grid = {'n_components':[5,15,30,45]}
        pls_hpt = PLSRegression()
        pls_hpt_f = GridSearchCV(estimator=pls_hpt, cv=pds, param_grid=pls_param_grid)
        pls_hpt_fit = pls_hpt_f.fit(X, Y)
        gc.collect()
        
    #Fit model
    X = df[(df['yyyymm'] // 100 <= train_year)].drop(columns=non_predictor)
    X = X.fillna(0)
    Y = df[(df['yyyymm'] // 100 <= train_year)]['excess_return'].to_frame()
    
    X_test = df[(df['yyyymm'] // 100 == test_year)].drop(columns=non_predictor)
    X_test = X_test.fillna(0)
    Y_test = df[(df['yyyymm'] // 100 == test_year)]['excess_return']
        
    pls = PLSRegression(**pls_hpt_f.best_params_)
    pls_fit = pls.fit(X, Y)
        
    #Export model
    filename = "pls_%s.pkl" % train_year
    with open(filename, 'wb') as fp:
        pickle.dump(pls, fp)

    #Prediction and R2
    pls_prediction = pls_fit.predict(X_test).reshape(len(Y_test),1)
    error_pred = pls_prediction-Y_test.values.reshape(len(Y_test),1)
    pls_R2 = 1-sum(error_pred*error_pred)/sum(Y_test.to_numpy()*Y_test.to_numpy())
    pls_R_sq_list = np.append(pls_R_sq_list,np.array(pls_R2))
       
    #Feature Importance
    # Extract feature names
    pls_feature_names = list(X.columns)
    # First calculate full model R_sq
    y_hat = pls_fit.predict(X).reshape(len(Y),1)
    error = y_hat-Y.values
    R_sq = 1-sum(error*error)/sum(Y.to_numpy()*Y.to_numpy())
    # Create a empty data for storing the feature importance
    pls_R_sq_loss_table = X.loc[0,pls_feature_names]
    pls_R_sq_loss_table[pls_feature_names] = 0  
        
    for pred in pls_feature_names:
        # We adjust the y_hat by removing the term corresponding feature x features weight
        pred_temp = X[pred]
        X[pred] = 0
        y_hat_new = pls_fit.predict(X).reshape(len(Y),1)
        X[pred] = pred_temp
        error_new = y_hat_new-Y.values.reshape(len(Y),1)
        R_sq_new = 1-sum(error_new*error_new)/sum(Y.to_numpy()*Y.to_numpy())
        #print(R_sq_new)
        # Compute loss in R_sq
        loss_in_R2 = R_sq - R_sq_new
        pls_R_sq_loss_table[pred] = loss_in_R2
        
    pls_feat_imp_temp = pls_R_sq_loss_table/sum(pls_R_sq_loss_table)
    pls_feat_imp[test_year] = pls_feat_imp_temp
    '''
    if test_year==1997:
        pls_feat_imp.to_csv("pls_feat_imp_1997.csv",index=True)
        pd.DataFrame(pls_R_sq_list).to_csv("pls_R_sq_list_1997.csv",index=False)
    if test_year==2007:
        pls_feat_imp.to_csv("pls_feat_imp_2007.csv",index=True)
        pd.DataFrame(pls_R_sq_list).to_csv("pls_R_sq_list_2007.csv",index=False)
    '''
    gc.collect()
pls_feat_imp.to_csv("pls_feat_imp.csv",index=True)
pd.DataFrame(pls_R_sq_list).to_csv("pls_R_sq_list.csv",index=False)


'''
GBT
'''
gbt_R_sq_list = []
gbt_feat_imp_package = pd.DataFrame()
gbt_feat_imp = pd.DataFrame()
for train_year in range(1974,2004):
    validation_year = train_year + 12 
    test_year = validation_year + 1
    non_predictor = ['yyyymm','RET','SHROUT','mve0','prc','permno','excess_return']
    
    # Hyper-parameter tuning (for every 5 years)
    if(train_year+1)%5 == 0:
        X = df[(df['yyyymm'] // 100 <= validation_year)].drop(columns=non_predictor)
        X = X.fillna(0)
        Y = df[(df['yyyymm'] // 100 <= validation_year)]['excess_return'].to_frame()
        number_of_train = sum(1 for yr in df['yyyymm'] if yr//100 <= train_year)
        split_index = np.repeat(0, len(X))
        split_index[0:number_of_train-1]=-1
        pds = PredefinedSplit(test_fold = split_index)
        gbt_param_dist = dict(max_depth=[1], n_estimators=dists.randint(low=1, high=1000), learning_rate=dists.uniform(loc=0.01,scale=0.09))
        gbt_hpt = GradientBoostingRegressor(random_state=0)
        gbt_hpt_f = RandomizedSearchCV(estimator=gbt_hpt, param_distributions=gbt_param_dist, n_iter=4, cv=pds, random_state=0)
        gbt_hpt_fit = gbt_hpt_f.fit(X, Y)
        gc.collect()
        
    #Fit model
    X = df[(df['yyyymm'] // 100 <= train_year)].drop(columns=non_predictor)
    X = X.fillna(0)
    Y = df[(df['yyyymm'] // 100 <= train_year)]['excess_return'].to_frame()
    
    X_test = df[(df['yyyymm'] // 100 == test_year)].drop(columns=non_predictor)
    X_test = X_test.fillna(0)
    Y_test = df[(df['yyyymm'] // 100 == test_year)]['excess_return']
        
    gbt = GradientBoostingRegressor(**gbt_hpt_f.best_params_)
    gbt_fit = gbt.fit(X, Y)
        
    #Export model
    filename = "gbt_%s.pkl" % train_year
    with open(filename, 'wb') as fp:
        pickle.dump(gbt, fp)

    #Prediction and R2
    gbt_prediction = gbt_fit.predict(X_test).reshape(len(Y_test),1)
    error_pred = gbt_prediction-Y_test.values.reshape(len(Y_test),1)
    gbt_R2 = 1-sum(error_pred*error_pred)/sum(Y_test.to_numpy()*Y_test.to_numpy())
    gbt_R_sq_list = np.append(gbt_R_sq_list,np.array(gbt_R2))
       
    #Feature Importance
    # Extract feature names
    gbt_feature_names = list(X.columns)
    # First calculate full model R_sq
    y_hat = gbt_fit.predict(X).reshape(len(Y),1)
    error = y_hat-Y.values
    R_sq = 1-sum(error*error)/sum(Y.to_numpy()*Y.to_numpy())
    # Create a empty data for storing the feature importance
    gbt_R_sq_loss_table = X.loc[0,gbt_feature_names]
    gbt_R_sq_loss_table[gbt_feature_names] = 0  
        
    for pred in gbt_feature_names:
        # We adjust the y_hat by removing the term corresponding feature x features weight
        pred_temp = X[pred]
        X[pred] = 0
        y_hat_new = gbt_fit.predict(X).reshape(len(Y),1)
        X[pred] = pred_temp
        error_new = y_hat_new-Y.values.reshape(len(Y),1)
        R_sq_new = 1-sum(error_new*error_new)/sum(Y.to_numpy()*Y.to_numpy())
        #print(R_sq_new)
        # Compute loss in R_sq
        loss_in_R2 = R_sq - R_sq_new
        gbt_R_sq_loss_table[pred] = loss_in_R2
        
    gbt_feat_imp_temp = gbt_R_sq_loss_table/sum(gbt_R_sq_loss_table)
    gbt_feat_imp[test_year] = gbt_feat_imp_temp
    
    '''
    if test_year==1997:
        gbt_feat_imp.to_csv("gbt_feat_imp_1997.csv",index=True)
        pd.DataFrame(gbt_R_sq_list).to_csv("gbt_R_sq_list_1997.csv",index=False)
    if test_year==2007:
        gbt_feat_imp.to_csv("gbt_feat_imp_2007.csv",index=True)
        pd.DataFrame(gbt_R_sq_list).to_csv("gbt_R_sq_list_2007.csv",index=False)
    '''
    gc.collect()
gbt_feat_imp.to_csv("gbt_feat_imp.csv",index=True)
pd.DataFrame(gbt_R_sq_list).to_csv("gbt_R_sq_list.csv",index=False)


''''
Random Forest
'''
rf_R_sq_list = []
rf_feat_imp_package = pd.DataFrame()
rf_feat_imp = pd.DataFrame()
for train_year in range(1974,2004):
    validation_year = train_year + 12 
    test_year = validation_year + 1
    non_predictor = ['yyyymm','RET','SHROUT','mve0','prc','permno','excess_return']
  
    X_test = df[(df['yyyymm'] // 100 == test_year)].drop(columns=non_predictor)
    X_test = X_test.fillna(0)
    Y_test = df[(df['yyyymm'] // 100 == test_year)]['excess_return']
    
    number_of_train = sum(1 for yr in df['yyyymm'] if yr//100 <= train_year)
    split_index = np.repeat(0, len(X))
    split_index[0:number_of_train-1]=-1
    pds = PredefinedSplit(test_fold = split_index)
    
    # Hyper-parameter tuning (for every 5 years)
    if(train_year+1)%5 == 0:
        X = df[(df['yyyymm'] // 100 <= validation_year)].drop(columns=non_predictor)
        X = X.fillna(0)
        Y = df[(df['yyyymm'] // 100 <= validation_year)]['excess_return'].to_frame()
        number_of_train = sum(1 for yr in df['yyyymm'] if yr//100 <= train_year)
        split_index = np.repeat(0, len(X))
        split_index[0:number_of_train-1]=-1
        pds = PredefinedSplit(test_fold = split_index)
        rf_param_grid = {'max_depth':[2,4], 'n_estimators':[300], 'max_features':[10,30]}
        rfreg_hpt = RandomForestRegressor()
        regreg_hpt_f = GridSearchCV(estimator=rfreg_hpt, cv=pds, param_grid=rf_param_grid)
        regreg_hpt_fit = regreg_hpt_f.fit(X, Y)
        gc.collect()
        
    # Fit model
    X = df[(df['yyyymm'] // 100 <= train_year)].drop(columns=non_predictor)
    X = X.fillna(0)
    Y = df[(df['yyyymm'] // 100 <= train_year)]['excess_return'].to_frame()
    
    X_test = df[(df['yyyymm'] // 100 == test_year)].drop(columns=non_predictor)
    X_test = X_test.fillna(0)
    Y_test = df[(df['yyyymm'] // 100 == test_year)]['excess_return']

    rfreg = RandomForestRegressor(**regreg_hpt_f.best_params_)
    rf_fit = rfreg.fit(X,Y)
    #rfregf.best_estimator_
    
    #Export model
    filename = "rf_%s.pkl" % train_year
    with open(filename, 'wb') as fp:
        pickle.dump(rfreg, fp)
    
    #Prediction and R2
    rf_prediction = rf_fit.predict(X_test).reshape(len(Y_test),1)
    error_pred = rf_prediction-Y_test.values.reshape(len(Y_test),1)
    rf_R2 = 1-sum(error_pred*error_pred)/sum(Y_test.to_numpy()*Y_test.to_numpy())
    rf_R_sq_list = np.append(rf_R_sq_list,np.array(rf_R2))
      
    #Feature Importance
    # Extract feature names
    rf_feature_names = list(X.columns)
    # First calculate full model R_sq
    y_hat =rf_fit.predict(X).reshape(len(Y),1)
    error = y_hat-Y.values
    R_sq = 1-sum(error*error)/sum(Y.to_numpy()*Y.to_numpy())
    # Create a empty data for storing the feature importance
    rf_R_sq_loss_table = X.loc[0,rf_feature_names]
    rf_R_sq_loss_table[rf_feature_names] = 0  
        
    for pred in rf_feature_names:
        # We adjust the y_hat by removing the term corresponding feature x features weight
        pred_temp = X[pred]
        X[pred] = 0
        y_hat_new = rf_fit.predict(X).reshape(len(Y),1)
        X[pred] = pred_temp
        error_new = y_hat_new-Y.values
        R_sq_new = 1-sum(error_new*error_new)/sum(Y.to_numpy()*Y.to_numpy())
        #print(R_sq_new)
        # Compute loss in R_sq
        loss_in_R2 = R_sq - R_sq_new
        rf_R_sq_loss_table[pred] = loss_in_R2
        
    rf_feat_imp_temp = rf_R_sq_loss_table/sum(rf_R_sq_loss_table)
    rf_feat_imp[test_year] = rf_feat_imp_temp
    
    '''
    if test_year==1997:
        rf_feat_imp.to_csv("rf_feat_imp_1997.csv",index=True)
        pd.DataFrame(rf_R_sq_list).to_csv("rf_R_sq_list_1997.csv",index=False)
    if test_year==2007:
        rf_feat_imp.to_csv("rf_feat_imp_2007.csv",index=True)
        pd.DataFrame(rf_R_sq_list).to_csv("rf_R_sq_list_2007.csv",index=False)
    '''
    gc.collect()
    
#rf_feat_imp_package.to_csv("rf_feat_imp_package.csv",index=True)
rf_feat_imp.to_csv("rf_feat_imp.csv",index=True)
pd.DataFrame(rf_R_sq_list).to_csv("rf_R_sq_list.csv",index=False)


'''
Plot the feat importance graph
'''
model_list=['PLS','PCR','RF','GBT']
file_name_list=['pls_feat_imp.csv','pcr_feat_imp.csv','rf_feat_imp.csv','gbt_feat_imp.csv']

for i in range(0,len(model_list)):
    feat_imp= pd.read_csv(file_name_list[i], sep=",")
    feat_imp.rename(columns={list(feat_imp)[0]:'feature'},inplace=True)
    feat_imp.set_index('feature', inplace=True)
    feat_imp['feat_imp']=feat_imp.mean(axis=1)
    feat_imp = feat_imp.sort_values(by='feat_imp', ascending=False)
    feat_imp_to_plot = feat_imp[0:20]['feat_imp'].sort_values(ascending=True)
    # Plot the feature importances in horizontal bar charts
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    ax.barh(list(feat_imp_to_plot.index),
            feat_imp_to_plot, 
            align = 'center', color = 'blue')
    ax.set_title(model_list[i])
