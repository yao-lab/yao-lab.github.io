import pandas as pd
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

import warnings

warnings.filterwarnings('ignore')


def Data_Process(df=None):
    arr = []

    def top(x):
        x = x.sort_values(by='mvel1')
        arr.append(x.iloc[-1001:-1, :])

    df.groupby('DATE').apply(top)
    df_top = pd.concat(arr)
    arr = []

    def bottom(x):
        x = x.sort_values(by='mvel1')
        arr.append(x.iloc[:1000, :])

    df.groupby('DATE').apply(bottom)
    df_bottom = pd.concat(arr)
    return df_top, df_bottom

def Data_Process2(df=None):
    arr = []

    def func(x):
        x = x.copy()
        x.interpolate(method='linear', axis=0, inplace=True)
        arr.append(x)

    df.groupby('permno').apply(func)
    df_final = pd.concat(arr)
    return df_final

def OLS_3(df_train=None, df_validation=None, df_oos=None):
    # df_train.dropna(how='any', axis=0, inplace=True)
    df_train.fillna(0, inplace=True)
    df_oos.fillna(0, inplace=True)
    model = sm.ols(formula='RET ~ mvel1+mom12m+bm', data=df_train)
    result = model.fit()
    y = result.params[0] + result.params[1] * df_oos['mvel1'] + \
        result.params[2] * df_oos['mom12m'] + result.params[3] * df_oos['bm']
    RSS = ((df_oos['RET'] - y) ** 2).sum()
    TSS = ((df_oos['RET']) ** 2).sum()
    R_sqrt = 1 - RSS / TSS
    return R_sqrt

def OLS(df_train=None, df_validation=None, df_oos=None):
    df_train.fillna(0, inplace=True)
    df_oos.dropna(0, inplace=True)
    train_y = df_train['RET'].values.reshape(-1, 1)
    train_x = df_train.iloc[:, 4:].values
    test_x = df_oos.iloc[:, 4:].values
    model=LinearRegression()
    model.fit(train_x,train_y)
    y=model.predict(test_x)
    RSS = ((df_oos['RET'].values.reshape(-1, 1) - y) ** 2).sum()
    TSS = ((df_oos['RET'].values.reshape(-1, 1)) ** 2).sum()
    R_sqrt = 1 - RSS / TSS
    return R_sqrt

def PLS(df_train=None, df_validation=None, df_oos=None):
    df_train.fillna(0, inplace=True)
    df_validation.fillna(0, inplace=True)
    df_oos.fillna(0, inplace=True)
    train_y = df_train['RET'].values.reshape(-1, 1)
    train_x = df_train.iloc[:, 4:].values
    validation_y = df_validation['RET'].values.reshape(-1, 1)
    validation_x = df_validation.iloc[:, 4:].values
    test_x = df_oos.iloc[:, 4:].values
    MSE=100000000000

    for i in range(1,10):
        model = PLSRegression(n_components=i)
        model.fit(train_x, train_y)
        predict_y = model.predict(validation_x)
        MSE_temp = ((validation_y - predict_y) ** 2.).sum()
        if MSE_temp<MSE:
            MSE=MSE_temp
            n_components = i
    print('NO. of components: ',n_components)
    y = model.predict(test_x)

    RSS = ((df_oos['RET'].values.reshape(-1, 1) - y) ** 2).sum()
    TSS = ((df_oos['RET'].values.reshape(-1, 1)) ** 2).sum()
    R_sqrt = 1 - RSS / TSS
    return R_sqrt

def PCR(df_train=None, df_validation=None, df_oos=None):
    df_train.fillna(0, inplace=True)
    df_validation.fillna(0, inplace=True)
    df_oos.fillna(0, inplace=True)
    train_y = df_train['RET'].values.reshape(-1, 1)
    train_x = df_train.iloc[:, 4:].values
    validation_y = df_validation['RET'].values.reshape(-1, 1)
    validation_x = df_validation.iloc[:, 4:].values
    test_x = df_oos.iloc[:, 4:].values
    MSE = 100000000000
    n_components=0
    for i in range(5,100,5):
        model_pca = PCA(n_components=i)
        model_pca.fit(train_x)
        train_x_reduction = model_pca.transform(train_x)
        validation_x_reduction = model_pca.transform(validation_x)
        model_reg = LinearRegression()
        model_reg.fit(train_x_reduction, train_y)
        predict_y = model_reg.predict(validation_x_reduction)
        MSE_temp = ((validation_y - predict_y) ** 2.).sum()
        if MSE_temp<MSE:
            MSE=MSE_temp
            n_components = i
    print('NO. of components: ', n_components)
    model_pca = PCA(n_components=n_components)
    model_pca.fit(train_x)
    train_x_reduction = model_pca.transform(train_x)
    test_x_reduction = model_pca.transform(test_x)
    model_reg = LinearRegression()
    model_reg.fit(train_x_reduction, train_y)
    y = model_reg.predict(test_x_reduction)
    RSS = ((df_oos['RET'].values.reshape(-1, 1) - y) ** 2).sum()
    TSS = ((df_oos['RET'].values.reshape(-1, 1)) ** 2).sum()
    R_sqrt = 1 - RSS / TSS
    return R_sqrt


def RF(df_train=None, df_validation=None, df_oos=None):
    df_train.fillna(0, inplace=True)
    df_validation.fillna(0, inplace=True)
    df_oos.fillna(0, inplace=True)
    train_y = df_train['RET'].values.reshape(-1, 1)
    train_x = df_train.iloc[:, 4:].values
    validation_y = df_validation['RET'].values.reshape(-1, 1)
    validation_x = df_validation.iloc[:, 4:].values
    test_x = df_oos.iloc[:, 4:].values
    MSE = 100000000000
    max_depth=0
    for i in range(1,6):
        model = RandomForestRegressor(max_depth=i)
        model.fit(train_x, train_y)
        predict_y = model.predict(validation_x).reshape(-1, 1)
        MSE_temp = ((validation_y - predict_y).flatten() ** 2).sum()
        if MSE_temp < MSE:
            MSE = MSE_temp
            max_depth = i
    print('NO. max_depth: ', max_depth)
    model=RandomForestRegressor(max_depth=max_depth)
    model.fit(train_x, train_y)
    y = model.predict(test_x).reshape(-1, 1)
    RSS = ((df_oos['RET'].values.reshape(-1, 1) - y).reshape(-1) ** 2).sum()
    TSS = ((df_oos['RET'].values.reshape(-1, 1)).reshape(-1) ** 2).sum()
    R_sqrt = 1 - RSS / TSS
    return R_sqrt




def NN1(df_train=None, df_validation=None, df_oos=None):
    df_train.fillna(0, inplace=True)
    df_oos.fillna(0, inplace=True)
    train_y = df_train['RET'].values.reshape(-1, 1)
    train_x = df_train.iloc[:, 4:].values
    test_x = df_oos.iloc[:, 4:].values
    model = MLPRegressor(hidden_layer_sizes=(32), activation='relu')
    model.fit(train_x, train_y)
    # predict_y = model.predict(validation_x).reshape(-1, 1)
    # MSE = ((validation_y - predict_y).reshape(-1) ** 2).sum()
    # print(MSE)
    y = model.predict(test_x).reshape(-1, 1)
    RSS = ((df_oos['RET'].values.reshape(-1, 1) - y).flatten() ** 2).sum()
    TSS = ((df_oos['RET'].values.reshape(-1, 1)).flatten() ** 2).sum()
    R_sqrt = 1 - RSS / TSS
    return R_sqrt
# NN1()

def NN2(df_train=None, df_validation=None, df_oos=None):
    df_train.fillna(0, inplace=True)
    df_oos.fillna(0, inplace=True)
    train_y = df_train['RET'].values.reshape(-1, 1)
    train_x = df_train.iloc[:, 4:].values
    test_x = df_oos.iloc[:, 4:].values
    model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu')
    model.fit(train_x, train_y)
    # predict_y = model.predict(validation_x).reshape(-1, 1)
    # MSE = ((validation_y - predict_y).reshape(-1) ** 2).sum()
    # print(MSE)
    y = model.predict(test_x).reshape(-1, 1)
    RSS = ((df_oos['RET'].values.reshape(-1, 1) - y).reshape(-1) ** 2).sum()
    TSS = ((df_oos['RET'].values.reshape(-1, 1)).reshape(-1) ** 2).sum()
    R_sqrt = 1 - RSS / TSS
    return R_sqrt
# NN2()

def NN3(df_train=None, df_validation=None, df_oos=None):
    df_train.fillna(0, inplace=True)
    df_oos.fillna(0, inplace=True)
    train_y = df_train['RET'].values.reshape(-1, 1)
    train_x = df_train.iloc[:, 4:].values
    test_x = df_oos.iloc[:, 4:].values
    model = MLPRegressor(hidden_layer_sizes=(32, 16, 8), activation='relu')
    model.fit(train_x, train_y)
    # predict_y = model.predict(validation_x).reshape(-1, 1)
    # MSE = ((validation_y - predict_y).flatten() ** 2).sum()
    # print(MSE)
    y = model.predict(test_x).reshape(-1, 1)
    RSS = ((df_oos['RET'].values.reshape(-1, 1) - y).flatten() ** 2).sum()
    TSS = ((df_oos['RET'].values.reshape(-1, 1)).flatten() ** 2).sum()
    R_sqrt = 1 - RSS / TSS
    return R_sqrt


# NN3()
def NN4(df_train=None, df_validation=None, df_oos=None):
    df_train.fillna(0, inplace=True)
    df_oos.fillna(0, inplace=True)
    train_y = df_train['RET'].values.reshape(-1, 1)
    train_x = df_train.iloc[:, 4:].values
    test_x = df_oos.iloc[:, 4:].values
    model = MLPRegressor(hidden_layer_sizes=(32, 16, 8, 4), activation='relu')
    model.fit(train_x, train_y)
    # predict_y = model.predict(validation_x).reshape(-1, 1)
    # MSE = ((validation_y - predict_y).flatten() ** 2).sum()
    # print(MSE)
    y = model.predict(test_x).reshape(-1, 1)
    RSS = ((df_oos['RET'].values.reshape(-1, 1) - y).flatten() ** 2).sum()
    TSS = ((df_oos['RET'].values.reshape(-1, 1)).flatten() ** 2).sum()
    R_sqrt = 1 - RSS / TSS
    # print('R_sqrt:', R_sqrt)
    return R_sqrt

def NN5(df_train=None, df_validation=None, df_oos=None):
    df_train.fillna(0, inplace=True)
    df_oos.fillna(0, inplace=True)
    train_y = df_train['RET'].values.reshape(-1, 1)
    train_x = df_train.iloc[:, 4:].values
    test_x = df_oos.iloc[:, 4:].values
    model = MLPRegressor(hidden_layer_sizes=(32, 16, 8, 4, 2), activation='relu')
    model.fit(train_x, train_y)
    # predict_y = model.predict(validation_x).reshape(-1, 1)
    # MSE = ((validation_y - predict_y).flatten() ** 2).sum()
    # print(MSE)
    y = model.predict(test_x).reshape(-1, 1)
    RSS = ((df_oos['RET'].values.reshape(-1, 1) - y).flatten() ** 2).sum()
    TSS = ((df_oos['RET'].values.reshape(-1, 1)).flatten() ** 2).sum()
    R_sqrt = 1 - RSS / TSS
    # print('R_sqrt:', R_sqrt)
    return R_sqrt

def ENet(df_train=None, df_validation=None, df_oos=None):
    df_train.fillna(0, inplace=True)
    df_validation.fillna(0, inplace=True)
    df_oos.fillna(0, inplace=True)
    train_y = df_train['RET'].values.reshape(-1, 1)
    train_x = df_train.iloc[:, 4:].values
    validation_y = df_validation['RET'].values.reshape(-1, 1)
    validation_x = df_validation.iloc[:, 4:].values
    test_x = df_oos.iloc[:, 4:].values
    MSE=100000000000
    alpha=0
    for i in range(0,100,5):
        print(i)
        model = ElasticNet(alpha=i)
        model.fit(train_x, train_y)
        predict_y = model.predict(validation_x).reshape(-1, 1)
        MSE_temp = ((validation_y - predict_y).flatten() ** 2).sum()
        if MSE_temp < MSE:
            MSE = MSE_temp
            alpha = i
    # predict_y = model.predict(validation_x).reshape(-1, 1)
    # MSE = ((validation_y - predict_y).reshape(-1) ** 2).sum()
    # print(MSE)
    print('Alpha: ', alpha)
    model=ElasticNet(alpha=alpha)
    model.fit(train_x, train_y)
    y = model.predict(test_x).reshape(-1, 1)
    RSS = ((df_oos['RET'].values.reshape(-1, 1) - y).reshape(-1) ** 2).sum()
    TSS = ((df_oos['RET'].values.reshape(-1, 1)).reshape(-1) ** 2).sum()
    R_sqrt = 1 - RSS / TSS
    return R_sqrt
