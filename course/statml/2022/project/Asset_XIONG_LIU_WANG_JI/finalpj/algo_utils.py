from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet,HuberRegressor
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
import pandas as pd
import lightgbm as lgb
# from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def r2_score(Y_test, Y_test_pred):
    Y_test_pred = Y_test_pred.reshape(Y_test_pred.shape[0], 1)
    RSS = ((Y_test- Y_test_pred) ** 2).sum()
    TSS = ((Y_test) ** 2).sum()
    R_sqrt = 1 - RSS / TSS
    # print(R_sqrt)
    return R_sqrt.values
### linear
def ols(X_train, Y_train,X_val,Y_val, X_test, Y_test,scaler='Raw' ):
    if scaler == 'Raw':
        pass
    elif scaler == 'Standard':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    fitter = LinearRegression(fit_intercept=False)
    fitter.fit(X_train, Y_train)
    Y_train_pred = fitter.predict(X_train)
    train_r2= r2_score(Y_train, Y_train_pred)
    Y_val_pred = fitter.predict(X_val)
    val_r2= r2_score(Y_val, Y_val_pred)
    Y_test_pred = fitter.predict(X_test)
    test_r2= r2_score(Y_test, Y_test_pred)
    print(train_r2, val_r2, test_r2)
    
    return train_r2, val_r2, test_r2, fitter.coef_

def ridge(X_train, Y_train,X_val,Y_val, X_test, Y_test,params,scaler='Raw'  ):
    if scaler == 'Raw':
        pass
    elif scaler == 'Standard':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    fitter = Ridge(**params)
    fitter.fit(X_train, Y_train)
    Y_train_pred = fitter.predict(X_train)
    train_r2= r2_score(Y_train, Y_train_pred)
    Y_val_pred = fitter.predict(X_val)
    val_r2= r2_score(Y_val, Y_val_pred)
    Y_test_pred = fitter.predict(X_test)
    test_r2= r2_score(Y_test, Y_test_pred)
    return train_r2, val_r2, test_r2, fitter.coef_

def lasso(X_train, Y_train,X_val,Y_val, X_test, Y_test,params,scaler='Raw'  ):
    if scaler == 'Raw':
        pass
    elif scaler == 'Standard':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    fitter = Lasso(**params)
    fitter.fit(X_train, Y_train)
    Y_train_pred = fitter.predict(X_train)
    train_r2= r2_score(Y_train, Y_train_pred)
    Y_val_pred = fitter.predict(X_val)
    val_r2= r2_score(Y_val, Y_val_pred)
    Y_test_pred = fitter.predict(X_test)
    test_r2= r2_score(Y_test, Y_test_pred)
    return train_r2, val_r2, test_r2, fitter.coef_


def enet(X_train, Y_train,X_val,Y_val, X_test, Y_test,params,scaler='Raw'  ):
    if scaler == 'Raw':
        pass
    elif scaler == 'Standard':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    fitter = ElasticNet(**params)
    fitter.fit(X_train, Y_train)
    Y_train_pred = fitter.predict(X_train)
    train_r2= r2_score(Y_train, Y_train_pred)
    Y_val_pred = fitter.predict(X_val)
    val_r2= r2_score(Y_val, Y_val_pred)
    Y_test_pred = fitter.predict(X_test)
    test_r2= r2_score(Y_test, Y_test_pred)
    return train_r2, val_r2, test_r2, fitter.coef_


def huber(X_train, Y_train,X_val,Y_val, X_test, Y_test,params,scaler='Raw'  ):
    if scaler == 'Raw':
        pass
    elif scaler == 'Standard':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    fitter = HuberRegressor(**params)
    fitter.fit(X_train, Y_train)
    Y_train_pred = fitter.predict(X_train)
    train_r2= r2_score(Y_train, Y_train_pred)
    Y_val_pred = fitter.predict(X_val)
    val_r2= r2_score(Y_val, Y_val_pred)
    Y_test_pred = fitter.predict(X_test)
    test_r2= r2_score(Y_test, Y_test_pred)
    
    return train_r2, val_r2, test_r2, fitter.coef_

#### dim reduction
def pls(X_train, Y_train,X_val,Y_val, X_test, Y_test,params,scaler='Raw'  ):
    # if scaler == 'Raw':
    #     pass
    # elif scaler == 'Standard':
    #     scaler = StandardScaler()
    #     scaler.fit(X_train)
    #     X_train = scaler.transform(X_train)
    #     X_val = scaler.transform(X_val)
    #     X_test = scaler.transform(X_test)
    fitter = PLSRegression(**params)
    fitter.fit(X_train, Y_train)
    Y_train_pred = fitter.predict(X_train)
    train_r2= r2_score(Y_train, Y_train_pred)
    Y_val_pred = fitter.predict(X_val)
    val_r2= r2_score(Y_val, Y_val_pred)
    Y_test_pred = fitter.predict(X_test)
    test_r2= r2_score(Y_test, Y_test_pred)
    return train_r2, val_r2, test_r2, fitter.coef_


def pcs(X_train, Y_train,X_val,Y_val, X_test, Y_test,params,scaler='Raw'  ):
    if scaler == 'Raw':
        pass
    elif scaler == 'Standard':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    dim_reduction = PCA(**params)
    dim_reduction.fit(X_train)
    X_train = dim_reduction.transform(X_train)
    X_val = dim_reduction.transform(X_val)
    X_test = dim_reduction.transform(X_test)
    fitter = LinearRegression()
    fitter.fit(X_train, Y_train)
    Y_train_pred = fitter.predict(X_train)
    train_r2= r2_score(Y_train, Y_train_pred)
    Y_val_pred = fitter.predict(X_val)
    val_r2= r2_score(Y_val, Y_val_pred)
    Y_test_pred = fitter.predict(X_test)
    test_r2= r2_score(Y_test, Y_test_pred)
    return train_r2, val_r2, test_r2, fitter.coef_



##### tree
def rf(X_train, Y_train,X_val,Y_val, X_test, Y_test,params,scaler='Raw'  ):
    if scaler == 'Raw':
        pass
    elif scaler == 'Standard':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    
    fitter = RandomForestRegressor(**params)
    fitter.fit(X_train, Y_train)
    Y_train_pred = fitter.predict(X_train)
    train_r2= r2_score(Y_train, Y_train_pred)
    Y_val_pred = fitter.predict(X_val)
    val_r2= r2_score(Y_val, Y_val_pred)
    Y_test_pred = fitter.predict(X_test)
    test_r2= r2_score(Y_test, Y_test_pred)
    return train_r2, val_r2, test_r2, fitter.feature_importances_

def gbdt(X_train, Y_train,X_val,Y_val, X_test, Y_test,params,scaler='Raw'  ):
    if scaler == 'Raw':
        pass
    elif scaler == 'Standard':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    
    fitter = GradientBoostingRegressor(**params)
    fitter.fit(X_train, Y_train)
    Y_train_pred = fitter.predict(X_train)
    train_r2= r2_score(Y_train, Y_train_pred)
    Y_val_pred = fitter.predict(X_val)
    val_r2= r2_score(Y_val, Y_val_pred)
    Y_test_pred = fitter.predict(X_test)
    test_r2= r2_score(Y_test, Y_test_pred)
    return train_r2, val_r2, test_r2, fitter.feature_importances_


def lgbr(X_train, Y_train,X_val,Y_val, X_test, Y_test,params,scaler='Raw'  ):
    if scaler == 'Raw':
        pass
    elif scaler == 'Standard':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    
    fitter = lgb.LGBMRegressor(**params)
    fitter.fit(X_train, Y_train)
    Y_train_pred = fitter.predict(X_train)
    train_r2= r2_score(Y_train, Y_train_pred)
    Y_val_pred = fitter.predict(X_val)
    val_r2= r2_score(Y_val, Y_val_pred)
    Y_test_pred = fitter.predict(X_test)
    test_r2= r2_score(Y_test, Y_test_pred)
    return train_r2, val_r2, test_r2, fitter.feature_importances_




#### nn
def nn(X_train, Y_train,X_val,Y_val, X_test, Y_test,params,scaler='Raw'  ):
    if scaler == 'Raw':
        pass
    elif scaler == 'Standard':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    
    fitter = MLPRegressor(**params)
    fitter.fit(X_train, Y_train)
    Y_train_pred = fitter.predict(X_train)
    train_r2= r2_score(Y_train, Y_train_pred)
    Y_val_pred = fitter.predict(X_val)
    val_r2= r2_score(Y_val, Y_val_pred)
    Y_test_pred = fitter.predict(X_test)
    test_r2= r2_score(Y_test, Y_test_pred)
    return train_r2, val_r2, test_r2, fitter.coefs_

