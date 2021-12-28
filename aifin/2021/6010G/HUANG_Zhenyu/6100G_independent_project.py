import gc
import inspect
import random
import time
import warnings
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from downcast import reduce
from numba import jit
from scipy.stats import norm, t
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, ElasticNet, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import acovf
from sklearn.inspection import permutation_importance

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from tune_sklearn import TuneGridSearchCV
import lightgbm as lgb
import xgboost as xgb


# Functions for DM Test
def dm_test(dm1, dm2, method='Harvey'):
    dm1 = [x for x in dm1 if np.isnan(x) == False]
    dm2 = [x for x in dm2 if np.isnan(x) == False]
    if len(dm1) != len(dm2):
        raise Exception('The lengths of the three inputs do not match')
    if len(dm1) == 0:
        raise Exception('The length of the input list should be more than 0')
    dm = np.array(dm2) - np.array(dm1)
    mean = np.mean(dm)
    n = len(dm)
    h1 = int(np.ceil(pow(n, 1 / 3))) + 1
    h = int(np.sign(h1)) * h1
    '''
    autocovariance = []
    for tau in range(n):
        temp = 0.0
        for i in range(tau, n):
            temp += (dm[i] - mean) * (dm[i - tau] - mean)
        autocovariance.append(temp / n)
    '''
    autocovariance = acovf(dm)
    std = (autocovariance[0] + 2 * np.sum(autocovariance[1:h])) / n
    if std <= 0:
        h = 1
        std = np.var(dm) / (n - 1)
    dm_stat = mean / np.sqrt(std)
    p_value = 2 * norm.cdf(-np.abs(dm_stat))
    if method == 'Harvey':
        harvey = (n + 1 - 2 * h + h * (h - 1)) / n
        dm_stat = dm_stat * np.sqrt(harvey)
        p_value = 2 * t.cdf(-abs(dm_stat), df=n - 1)
    return dm_stat, p_value


def dm_comparison(dm1, dm2, dm1_name, dm2_name):
    if dm1.shape[1] != dm2.shape[1]:
        raise Exception('The lengths of the three inputs do not match')
    if dm1.shape[1] == 0:
        raise Exception('The length of the input list should be more than 0')
    dm_result = []
    p_value = []
    df = pd.DataFrame()
    for i in range(dm1.shape[1]):
        dm_result.append(dm_test(dm1.iloc[:, i], dm2.iloc[:, i])[0])
        p_value.append(dm_test(dm1.iloc[:, i], dm2.iloc[:, i])[1])
    name = dm2_name + '-' + dm1_name
    df[name] = dm_result
    df[name + '_p_value'] = p_value
    return df


@jit
def reduce_mem_usage(df, verbose=True, method='downcast'):
    starts = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    if method == 'downcast':
        df = reduce(df)
    else:
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction) and time is '.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem) + str(time.time() - starts))
    return df


def get_variable_list(data, name):
    variable_list = data[name]
    all_variable = list(set(data[name]))
    all_variable.sort()
    return variable_list, all_variable


@jit
def missing_data_check(data, period):
    missing_data = pd.DataFrame()
    missing_data.index = data.columns
    _, all_date = get_variable_list(data, period)
    for j in all_date:
        begin = time.time()
        temp = data[data[period] == j]
        percent = temp.isnull().sum() / temp.isnull().count()
        missing_data['{}'.format(str(j))] = percent.values
        last = time.time()
        print('Finished checking year ' + str(j) + ' and time is {}s'.format(str(last - begin)))
    return missing_data


def one_hot_encoder(df):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


@jit
def monthly_imputation(data):
    _, all_month = get_variable_list(data, 'month')
    new_data = pd.DataFrame()
    all_month = [month for month in all_month if month <= 198412]
    for month in all_month:
        a = time.time()
        file = data[data['month'] == month]
        percent = file.isnull().sum() / file.isnull().count()
        percent = percent.to_frame()
        col = percent.loc[percent.values == 1]
        col_fill = col.index.to_list()
        file[col_fill] = 0.0
        new_data = pd.concat([new_data, file])
        print('Finished filling blanks for month ' + str(month) + ' and time is {}s'.format(str(time.time() - a)))
    temp = data[data['year'] >= 1985]
    new_data = pd.concat([new_data, temp])
    return new_data


def drop_low_observation(data, fraction=0.1):
    data['observation'] = data.groupby('permno')['permno'].transform('count')
    data = data[data['observation'] >= int(np.floor(data['observation'].quantile(fraction)))]
    data = data.drop(columns='observation')
    return data


def create_rolling(data, rolling_col):
    start_time = time.time()
    label = ['permno', 'month', 'sic2'] + rolling_col
    temp = data[label]
    print('Create rolling aggs')

    for TARGET in rolling_col:
        begin = time.time()
        print('Creating rolling feature for', TARGET)
        for d_shift in [1, 3, 6]:
            time1 = time.time()
            print('Shifting period:', d_shift)
            for d_window in [3, 6, 12]:
                time2 = time.time()
                col_name_mean = 'rolling_mean_' + TARGET + '_' + str(d_shift) + '_' + str(d_window)
                col_name_std = 'rolling_median_' + TARGET + '_' + str(d_shift) + '_' + str(d_window)
                temp[col_name_mean] = temp.groupby(['permno'])[TARGET].transform(
                    lambda x: x.shift(d_shift).rolling(d_window).mean())
                temp[col_name_std] = temp.groupby(['permno'])[TARGET].transform(
                    lambda x: x.shift(d_shift).rolling(d_window).std())
                print('Finished Shifting window ' + str(d_window) + ' and time is {}s'.format(str(time.time() - time2)))
            print('Finished Shifting period ' + str(d_shift) + ' and time is {}s'.format(str(time.time() - time1)))
        print('Finished Shifting feature ' + TARGET + ' and time is {}s'.format(str(time.time() - begin)))

    print('%0.2f min: Lags' % ((time.time() - start_time) / 60))
    return temp


def create_lag(data):
    start_time = time.time()
    LAG_MONTHS = [1, 3, 6, 12]
    temp = data[['permno', 'month', 'sic2', 'RET']]
    temp = temp.assign(**{
        '{}_lag_{}'.format(col, l): data.groupby(['permno'])[col].transform(lambda x: x.shift(l))
        for l in LAG_MONTHS
        for col in ['RET']
    })
    print('Finished creating lags and time is {}s'.format(time.time() - start_time))
    return temp


def data_processing(data, data1, list_to_drop, follow_paper=False, keep_na=False):
    if follow_paper:
        data = data.loc[(data['year'] >= 1957) & (data['year'] <= 2016)].reset_index(drop=True)
        data1 = data1.loc[(data1['year'] >= 1957) & (data1['year'] <= 2016)].reset_index(drop=True)
    else:
        data = data.loc[(data['month'] >= 197506) & (data['month'] <= 201906)].reset_index(drop=True)
        data1 = data1.loc[(data1['yyyymm'] >= 197506) & (data1['yyyymm'] <= 201906)].reset_index(drop=True)

    # Generate macroeconomic characteristics
    data1['tms'] = data1['lty'] - data1['tbl']
    data1['dfy'] = data1['BAA'] - data1['AAA']
    data1['d/p'] = data1['D12'].apply(np.log) - data1['Index'].apply(np.log)
    data1['ep_marco'] = data1['E12'].apply(np.log) - data1['Index'].apply(np.log)
    data1 = data1[['yyyymm', 'd/p', 'ep_marco', 'b/m', 'ntis', 'tbl', 'tms', 'dfy', 'svar']]

    industry_dummy = data['sic2']
    permno = data['permno']
    month = data['month']
    age = data['age']
    data = data.drop(columns=list_to_drop)
    print('Begin to process data')
    start1 = time.time()
    temp = pd.concat([permno, industry_dummy], join="inner", axis=1)
    temp = temp.groupby(['permno']).transform(lambda x: x.fillna(x.median()))
    print('Finished step 1 and time is {}s'.format(str(time.time() - start1)))
    if keep_na:
        data = pd.concat([data, temp, age], join="inner", axis=1)
    else:
        age = age.fillna(0)
        print('Finished step 2 and time is {}s'.format(str(time.time() - start1)))
        data = monthly_imputation(data)
        data = pd.concat([data, temp], join="inner", axis=1)
        data = data.groupby(['month', 'sic2']).transform(lambda x: x.fillna(method='bfill')).fillna(method='ffill')
        print('Finished imputing by sector and time is {}s'.format(str(time.time() - start1)))
        data = pd.concat([data, month], join="inner", axis=1)
        data = data.groupby(['month']).transform(lambda x: x.fillna(method='bfill')).fillna(method='ffill')
        print('Finished imputing missing values and time is {}s'.format(str(time.time() - start1)))
        data = pd.concat([month, temp, age, data], join="inner", axis=1)
        data = data.dropna(axis=0)
    del temp, industry_dummy, permno, age, month
    gc.collect()
    data = data.merge(data1, how='inner', left_on='month', right_on='yyyymm')
    data = data.drop(columns='yyyymm')
    print('Data processing finished and time is {}'.format(str(time.time() - start1)))
    return data.reset_index(drop=True)


@jit
def interaction_feature(data, macro_feature, na_contain=False):
    data = data.drop(columns=['permno', 'month', 'year', 'mve0', 'SHROUT', 'prc', 'sic2', 'age'])
    data = data.drop(columns=['convind', 'divo', 'rd', 'securedind', 'divi'])
    temp = data[macro_feature]
    data = data.drop(columns=macro_feature)
    if na_contain:
        features_selected = interaction_variable
        data = pd.concat([data, temp], join='inner', axis=1)
    else:
        corr_on = time.time()
        corr_matrix = data.corr()
        corr_matrix = abs(corr_matrix)
        print('Finished calculating correlation matrix and time is {}'.format(str(time.time() - corr_on)))
        num_variables_to_keep = int(np.ceil(data.shape[1] / 3))
        features_selected = corr_matrix["RET"].sort_values(ascending=False).index.drop("RET")[:num_variables_to_keep]
        data = data[features_selected]
        data = pd.concat([data, temp], join='inner', axis=1)
    new_data = pd.DataFrame()
    for x in temp:
        start1 = time.time()
        for y in features_selected:
            start2 = time.time()
            name = str(x) + '*' + str(y)
            new_data[name] = data[x] * data[y]
            print('Finished calculating ' + str(y) + ' and time is {}'.format(str(time.time() - start2)))
        print('Finished calculating ' + str(x) + ' and time is {}'.format(str(time.time() - start1)))

    new_data = reduce_mem_usage(new_data)
    return new_data, features_selected


# Data Standardization
def data_std(data, stds, min_max, indicator_label, dum_label, int_label, train=True):
    if not dum_label:
        int_label = int_label + ['sic2']
    if 'age' not in data.columns:
        int_label = ['sic2']
    int_data = data[int_label]
    drop_label = indicator_label + dum_label + int_label
    keep_label = indicator_label + dum_label
    temp = data[keep_label]
    data = data.drop(columns=drop_label)
    org_col = data.columns
    if train:
        standard = stds.fit_transform(data)
        integer = min_max.fit_transform(int_data)
    else:
        standard = stds.transform(data)
        integer = min_max.transform(int_data)
    df_std = pd.DataFrame(data=standard, columns=org_col)
    df_int = pd.DataFrame(data=integer, columns=int_label)
    new_data = pd.concat([temp, df_std, df_int], join="inner", axis=1)
    del temp, df_std, standard, df_int
    gc.collect()
    return new_data.reset_index(drop=True)


def get_feature_importance(indicator, model, present_year, default_year):
    if indicator == 'linear':
        importance = [float(x) for x in model.coef_]
        temp = pd.DataFrame(
            data={'feature': model.feature_names_in_, 'importance{}'.format(str(
                present_year - default_year + 1)): np.abs(importance)}).sort_values('feature').reset_index(drop=True)
    elif indicator == 'pca':
        temp = pd.DataFrame(
            data={'feature': model.feature_names_in_, 'importance{}'.format(str(
                present_year - default_year + 1)): np.abs(model.steps[1][1].coef_)}).sort_values('feature').reset_index(
            drop=True)
    elif indicator == 'tree':
        temp = pd.DataFrame(
            data={'feature': model.feature_names_in_, 'importance{}'.format(str(
                present_year - default_year + 1)): model.feature_importances_}).sort_values('feature').reset_index(
            drop=True)
    else:
        temp = None
    return temp


def retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


def r2_score_df(score_ord, score_adj, top_ord, top_adj, bottom_ord, bottom_adj, name):
    if not top_ord:
        df = pd.DataFrame(data=[score_ord, score_adj],
                          index=[retrieve_name(score_ord), retrieve_name(score_adj)])
        df = df.T
        df.columns = name[:2]
        df['change'] = df.iloc[:, 1] - df.iloc[:, 0]
    else:
        df = pd.DataFrame(data=[score_ord, score_adj, top_ord, top_adj, bottom_ord, bottom_adj],
                          index=[retrieve_name(score_ord), retrieve_name(score_adj), retrieve_name(top_ord),
                                 retrieve_name(top_adj), retrieve_name(bottom_ord), retrieve_name(bottom_adj)])
        df = df.T
        df.columns = name
        df['change'] = df.iloc[:, 1] - df.iloc[:, 0]
        df['change_top'] = df.iloc[:, 3] - df.iloc[:, 2]
        df['change_bottom'] = df.iloc[:, 5] - df.iloc[:, 4]

    return df


def get_model_name(model, model_type):
    if model_type == 'pca':
        return str(model.steps[0][1])[:str(model.steps[0][1]).index('(')] + ' with ' + str(
            model.steps[1][1])[:str(model.steps[1][1]).index('(')]
    elif model_type == 'network':
        return 'NN with {} hidden layer(s)'.format(str(len(model.hidden_layer_sizes)))
    else:
        return str(model)[:str(model).index('(')]


def r2_score_demean(y_true, y_pred):
    rss = np.sum([(a - b) ** 2 for a, b in zip(y_true, y_pred)])
    tss = np.sum([a ** 2 for a in y_true])
    return 1 - rss / tss


def adjusted_r2_demean(y_true, y_pred, data):
    r2 = r2_score_demean(y_true, y_pred)
    n = len(data)
    p = data.shape[1]
    coef = (n - 1) / (n - p - 1)
    return 1 - (1 - r2) * coef


@jit
def get_model_error(x, y, model, year, year_list, m_type):
    begin = time.time()
    errors = []
    data = pd.concat([x, y, year], join="inner", axis=1)
    for years in year_list:
        temp = data[data['year'] == years].reset_index(drop=True)
        y_temp = np.array(temp['RET']).reshape(-1, 1)
        temp = temp.drop(columns=['year', 'RET'])
        if m_type == 'xgb':
            data_temp = xgb.DMatrix(temp, label=y_temp)
            y_pred = model.predict(data_temp)
        else:
            y_pred = model.predict(temp)
        y_pred = y_pred.reshape(-1, 1)
        error = mean_squared_error(y_temp, y_pred)
        errors.append(error)
    finish = time.time()
    print('Finished obtaining model error and time is {}s'.format(str(finish - begin)))
    return errors


@jit
def period_prediction(x, y, model, date, date_list, period, m_type, random_sample=False, fraction=0.5, training=True):
    score_ord = []
    score_adj = []
    data = pd.concat([x, y, date], join="inner", axis=1)
    if random_sample:
        timestamp = random.sample(date_list, int(np.ceil(len(date_list) * fraction)))
    else:
        timestamp = date_list
    for timetick in timestamp:
        temp = data[data[period] == timetick].reset_index(drop=True)
        y_temp = temp['RET']
        temp = temp.drop(columns=[period, 'RET'])
        if m_type == 'xgb':
            data_temp = xgb.DMatrix(temp, label=y_temp)
            y_pred = model.predict(data_temp)
        else:
            y_pred = model.predict(temp)
        score_adj.append(r2_score_demean(y_temp, y_pred))
        score_ord.append(r2_score(y_temp, y_pred))
    if training:
        return np.mean(score_ord), np.mean(score_adj)
    return pd.Series(score_ord), pd.Series(score_adj)


# top and bottom stocks sorted by market value in each month or year
@jit
def top_bottom_score(x, mv, y, date, date_list, model, m_type, flag='bottom', time_range='year'):
    score_ord = []
    score_adj = []
    newdata = pd.concat([x, mv, y, date], join="inner", axis=1)
    for date in date_list:
        temp = newdata[newdata[time_range] == date].reset_index(drop=True)
        temp = temp.sort_values('mve0', ascending=False).reset_index(drop=True)
        if flag == 'top':
            temp = temp.iloc[:1000]
        else:
            temp = temp.iloc[-1000:]
        y_temp = temp['RET']
        temp = temp.drop(columns=[time_range, 'RET', 'mve0'])
        if m_type == 'xgb':
            data_temp = xgb.DMatrix(temp, label=y_temp)
            y_pred = model.predict(data_temp)
        else:
            y_pred = model.predict(temp)
        score_adj.append(r2_score_demean(y_temp, y_pred))
        score_ord.append(r2_score(y_temp, y_pred))

    return np.mean(score_ord), np.mean(score_adj)


@jit
def model_training(data, list_drop, model, model_type, top_bottom=True, industry_encoder=False, standardization=True,
                   follow_paper=False, get_importance=True, prediction='year', start_year=1990, end_year=2011):
    valid_score_ord = []
    valid_score_adj = []
    test_score_ord = []
    test_score_adj = []
    valid_top_ord = []
    valid_top_adj = []
    valid_bottom_ord = []
    valid_bottom_adj = []
    test_top_ord = []
    test_top_adj = []
    test_bottom_ord = []
    test_bottom_adj = []
    models = []
    valid_dm = pd.DataFrame()
    test_dm = pd.DataFrame()
    features = pd.DataFrame()
    col = [x for x in data.columns if x not in list_drop]
    col.sort()
    features['feature'] = col
    if model_type == 'xgb':
        model_name = 'xgboost'
    elif model_type == 'lgbm':
        model_name = 'LGBM'
    else:
        model_name = get_model_name(model, model_type)
        if model_name == 'SGDRegressor':
            model_name = 'Enet+H'
    if industry_encoder:
        str_num = [str(num) for num in data['sic2']]
        data['sic2'] = str_num
        data, industry_col = one_hot_encoder(data)
    else:
        industry_col = []
    if follow_paper:
        valid_time = 1976
        end_time = 2007
        time_period = 12
    else:
        valid_time = 1990
        end_time = 2011
        time_period = 8
    for i in range(max(start_year, valid_time), min(end_year, end_time)):
        start1 = time.time()
        test_time = i + time_period
        # train, valid, test sample splitting
        print('Start training No. ' + str(i - valid_time + 1) + ' and training sample before year ' + str(i))
        data_train = data[data['year'] < i].copy().reset_index(drop=True)
        data_valid = data.loc[(data['year'] >= i) & (data['year'] < test_time)].copy().reset_index(drop=True)
        data_test = data[data['year'] >= test_time].copy().reset_index(drop=True)
        valid_month, all_valid_month = get_variable_list(data_valid, 'month')
        test_month, all_test_month = get_variable_list(data_test, 'month')
        valid_year, all_valid_year = get_variable_list(data_valid, 'year')
        test_year, all_test_year = get_variable_list(data_test, 'year')
        y_train = data_train['RET']
        data_train = data_train.drop(columns=list_drop)
        y_valid = data_valid['RET']
        mv_valid = data_valid['mve0']
        data_valid = data_valid.drop(columns=list_drop)
        y_test = data_test['RET']
        mv_test = data_test['mve0']
        data_test = data_test.drop(columns=list_drop)
        if standardization:
            ss = StandardScaler()
            mm = MinMaxScaler()
            t0 = time.time()
            data_train = data_std(data_train, ss, mm, indicators, industry_col, int_col)
            data_valid = data_std(data_valid, ss, mm, indicators, industry_col, int_col, train=False)
            data_test = data_std(data_test, ss, mm, indicators, industry_col, int_col, train=False)
            t1 = time.time()
            print('Finished standardization no.' + str(i - valid_time + 1) + ' and time is {}s'.format(str(t1 - t0)))

        # model fitting
        tick1 = time.time()

        if model_type == 'lgbm':
            train_data = model.Dataset(data_train, label=y_train)
            valid_data = model.Dataset(data_valid, label=y_valid)
            reg = model.train(lgb_params, train_data, valid_sets=[valid_data, train_data], early_stopping_rounds=40,
                              verbose_eval=20)
        elif model_type == 'xgb':
            train_data = model.DMatrix(data_train, label=y_train)
            valid_data = model.DMatrix(data_valid, label=y_valid)
            eval_list = [(train_data, 'train'), (valid_data, 'valid')]
            reg = model.train(xgb_params, train_data, num_boost_round=1000, evals=eval_list, early_stopping_rounds=40,
                              verbose_eval=20)
        else:
            reg = model.fit(data_train, y_train)

        models.append(reg)
        tick2 = time.time()
        print('Finished model fitting no.' + str(i - valid_time + 1) + ' and time is {}s'.format(str(tick2 - tick1)))

        if get_importance:
            click = time.time()
            if model_type == 'lgbm':
                temp_feature = pd.DataFrame(data={'feature': reg.feature_name(), 'importance{}'.format(
                    str(i - valid_time + 1)): reg.feature_importance()}).sort_values('feature').reset_index(drop=True)
            elif model_type == 'xgb':
                important_feature = reg.get_score(importance_type='weight')
                keys = list(important_feature.keys())
                values = list(important_feature.values())
                df = pd.DataFrame()
                df['feature'] = keys
                df['importance{}'.format(str(i - valid_time + 1))] = values
                temp_feature = pd.DataFrame(data={'feature': data_train.columns})
                temp_feature = temp_feature.merge(df, on='feature')
            else:
                temp_feature = get_feature_importance(model_type, reg, i, valid_time)
            temp_feature = temp_feature.fillna(0)

            features = features.merge(temp_feature, on='feature')
            final = time.time()
            print('No.' + str(i - valid_time + 1) + ' Feature importance sorted and time is {}s'.format(
                str(final - click)))

        # Prediction and r2 score calculation
        predict_on = time.time()
        if prediction == 'month':
            valid_score_ord.append(period_prediction(
                data_valid, y_valid, reg, valid_month, all_valid_month, period=prediction, m_type=model_type)[0])
            valid_score_adj.append(period_prediction(
                data_valid, y_valid, reg, valid_month, all_valid_month, period=prediction, m_type=model_type)[1])
            test_score_ord.append(period_prediction(
                data_test, y_test, reg, test_month, all_test_month, period=prediction, m_type=model_type)[0])
            test_score_adj.append(period_prediction(
                data_test, y_test, reg, test_month, all_test_month, period=prediction, m_type=model_type)[1])
        elif prediction == 'year':
            valid_score_ord.append(period_prediction(
                data_valid, y_valid, reg, valid_year, all_valid_year, period=prediction, m_type=model_type)[0])
            valid_score_adj.append(period_prediction(
                data_valid, y_valid, reg, valid_year, all_valid_year, period=prediction, m_type=model_type)[1])
            test_score_ord.append(period_prediction(
                data_test, y_test, reg, test_year, all_test_year, period=prediction, m_type=model_type)[0])
            test_score_adj.append(period_prediction(
                data_test, y_test, reg, test_year, all_test_year, period=prediction, m_type=model_type)[1])
        else:
            y_pred_valid = reg.predict(data_valid)
            valid_score_adj.append(r2_score_demean(y_valid, y_pred_valid))
            valid_score_ord.append(r2_score(y_valid, y_pred_valid))
            y_pred_test = reg.predict(data_test)
            test_score_adj.append(r2_score_demean(y_test, y_pred_test))
            test_score_ord.append(r2_score(y_valid, y_pred_valid))

        print('Finished prediction no.' + str(i - valid_time + 1) + ' and time is {}s'.format(
            str(time.time() - predict_on)))

        # Get annual prediction error for DM calculation
        error_on = time.time()
        error_valid = get_model_error(data_valid, y_valid, reg, valid_year, all_valid_year, m_type=model_type)
        error_test = get_model_error(data_test, y_test, reg, test_year, all_test_year, m_type=model_type)
        print('Finished error calculation no.' + str(i - valid_time + 1) + ' and time is {}s'.format(
            str(time.time() - error_on)))

        # Prediction on top and bottom 1000 stocks ranked by their market values
        if top_bottom:
            top = time.time()
            valid_bottom_ord.append(top_bottom_score(
                data_valid, mv_valid, y_valid, valid_year, all_valid_year, reg, m_type=model_type)[0])
            valid_bottom_adj.append(top_bottom_score(
                data_valid, mv_valid, y_valid, valid_year, all_valid_year, reg, m_type=model_type)[1])
            test_bottom_ord.append(top_bottom_score(
                data_test, mv_test, y_test, test_year, all_test_year, reg, m_type=model_type)[0])
            test_bottom_adj.append(top_bottom_score(
                data_test, mv_test, y_test, test_year, all_test_year, reg, m_type=model_type)[1])

            valid_top_ord.append(top_bottom_score(
                data_valid, mv_valid, y_valid, valid_year, all_valid_year, reg, m_type=model_type, flag='top')[0])
            valid_top_adj.append(top_bottom_score(
                data_valid, mv_valid, y_valid, valid_year, all_valid_year, reg, m_type=model_type, flag='top')[1])
            test_top_ord.append(top_bottom_score(
                data_test, mv_test, y_test, test_year, all_test_year, reg, m_type=model_type, flag='top')[0])
            test_top_adj.append(top_bottom_score(
                data_test, mv_test, y_test, test_year, all_test_year, reg, m_type=model_type, flag='top')[1])
            bottom = time.time()
            print('Finished top_bottom calculation no.' + str(i - valid_time + 1) + ' and time is {}s'.format(
                str(bottom - top)))

        # Get series data for DM calculation
        test_dm['Training{}'.format(str(i - valid_time + 1))] = pd.Series(error_test)
        valid_dm['Training{}'.format(str(i - valid_time + 1))] = pd.Series(error_valid)
        end1 = time.time()
        print('Finished training no.' + str(i - valid_time + 1) + ' on ' + model_name + ' and time is {}s'.format(
            str(end1 - start1)))
        gc.collect()

    if get_importance:
        feature_importance = pd.DataFrame(
            data={'feature': features['feature'], 'importance': features.iloc[:, 1:].mean(axis=1)})
    else:
        feature_importance = None

    # Save r2 scores
    valid_score = r2_score_df(valid_score_ord, valid_score_adj, valid_top_ord, valid_top_adj, valid_bottom_ord,
                              valid_bottom_adj, column)
    test_score = r2_score_df(test_score_ord, test_score_adj, test_top_ord, test_top_adj, test_bottom_ord,
                             test_bottom_adj, column)
    return feature_importance, valid_score, test_score, models, model_name, valid_dm, test_dm


@jit
def predict_by_time(data, model, tick, list_drop, standardization=True, follow_paper=False, industry_encoder=False):
    valid_scores = pd.DataFrame()
    test_scores = pd.DataFrame()
    if follow_paper:
        valid_time = 1976
        time_period = 12
    else:
        valid_time = 1990
        time_period = 8
    if industry_encoder:
        str_num = [str(num) for num in data['sic2']]
        data['sic2'] = str_num
        data, industry_col = one_hot_encoder(data)
    else:
        industry_col = []
    for i in range(len(model)):
        print('Start prediction NO.' + str(i + 1))
        start = time.time()
        valid_tick = valid_time + i
        test_tick = time_period + valid_tick
        data_train = data[data['year'] < valid_tick].copy().reset_index(drop=True)
        data_valid = data.loc[(data['year'] >= valid_tick) & (data['year'] < test_tick)].copy().reset_index(drop=True)
        data_test = data[data['year'] >= test_tick].copy().reset_index(drop=True)
        valid_year, all_valid_year = get_variable_list(data_valid, 'year')
        valid_month, all_valid_month = get_variable_list(data_valid, 'month')
        test_year, all_test_year = get_variable_list(data_test, 'year')
        test_month, all_test_month = get_variable_list(data_test, 'month')
        data_train = data_train.drop(columns=list_drop)
        y_valid = data_valid['RET']
        data_valid = data_valid.drop(columns=list_drop)
        y_test = data_test['RET']
        data_test = data_test.drop(columns=list_drop)

        if standardization:
            ss = StandardScaler()
            mm = MinMaxScaler()
            t0 = time.time()
            data_train = data_std(data_train, ss, mm, indicators, industry_col, int_col)
            data_valid = data_std(data_valid, ss, mm, indicators, industry_col, int_col, train=False)
            data_test = data_std(data_test, ss, mm, indicators, industry_col, int_col, train=False)
            t1 = time.time()
            print('Finished standardization no.' + str(i + 1) + ' and time is {}s'.format(str(t1 - t0)))
        if tick == 'month':
            valid_scores[str(i) + '_month_ord_{}'.format(str(valid_tick))] = period_prediction(
                data_valid, y_valid, model[i], valid_month, all_valid_month, period=tick, training=False)[0]
            valid_scores[str(i) + '_month_adj_{}'.format(str(valid_tick))] = period_prediction(
                data_valid, y_valid, model[i], valid_month, all_valid_month, period=tick, training=False)[1]
            test_scores[str(i) + '_month_ord_{}'.format(str(test_tick))] = period_prediction(
                data_test, y_test, model[i], test_month, all_test_month, period=tick, training=False)[0]
            test_scores[str(i) + '_month_adj_{}'.format(str(test_tick))] = period_prediction(
                data_test, y_test, model[i], test_month, all_test_month, period=tick, training=False)[1]
        else:
            valid_scores[str(i) + '_year_ord_{}'.format(str(valid_tick))] = period_prediction(
                data_valid, y_valid, model[i], valid_year, all_valid_year, period=tick, training=False)[0]
            valid_scores[str(i) + '_year_adj_{}'.format(str(valid_tick))] = period_prediction(
                data_valid, y_valid, model[i], valid_year, all_valid_year, period=tick, training=False)[1]
            test_scores[str(i) + '_year_ord_{}'.format(str(test_tick))] = period_prediction(
                data_test, y_test, model[i], test_year, all_test_year, period=tick, training=False)[0]
            test_scores[str(i) + '_year_adj_{}'.format(str(test_tick))] = period_prediction(
                data_test, y_test, model[i], test_year, all_test_year, period=tick, training=False)[1]
        print('Finished prediction NO. ' + str(i + 1) + ' and time is {}'.format(str(time.time() - start)))

    return valid_scores, test_scores


@jit
def get_model_importance(data, list_drop, model, standardization=True, follow_paper=False):
    if follow_paper:
        valid_time = 1976
    else:
        valid_time = 1990
    features = pd.DataFrame()
    col = [x for x in data.columns if x not in list_drop]
    col.sort()
    features['feature'] = col
    industry_col = []
    for i in range(len(model)):
        print('Start feature obtaining NO.' + str(i + 1))
        valid_tick = valid_time + i
        data_train = data[data['year'] < valid_tick].copy().reset_index(drop=True)
        y_train = data_train['RET']
        data_train = data_train.drop(columns=list_drop)

        if standardization:
            ss = StandardScaler()
            mm = MinMaxScaler()
            t0 = time.time()
            data_train = data_std(data_train, ss, mm, indicators, industry_col, int_col)
            t1 = time.time()
            print('Finished standardization no.' + str(i + 1) + ' and time is {}s'.format(str(t1 - t0)))

        feature_on = time.time()
        result = permutation_importance(model[i], data_train, y_train, n_repeats=3, n_jobs=-1,
                                        scoring='neg_root_mean_squared_error', max_samples=0.5)
        temp = pd.DataFrame(data={'feature': data_train.columns, 'importance{}'.format(str(
            i + 1)): result.importances_mean}).sort_values('feature').reset_index(drop=True)
        features = features.merge(temp, on='feature')

        print('Finished getting feature importance no.' + str(i + 1) + ' and time is {}s'.format(str(
            time.time() - feature_on)))

    feature_importance = pd.DataFrame(
        data={'feature': features['feature'], 'importance': features.iloc[:, 1:].mean(axis=1)})

    return feature_importance


@jit
def get_dm_value(data, list_drop, model, industry_encoder=False, standardization=True, follow_paper=False):
    valid_dm = pd.DataFrame()
    test_dm = pd.DataFrame()
    if industry_encoder:
        str_num = [str(num) for num in data['sic2']]
        data['sic2'] = str_num
        data, industry_col = one_hot_encoder(data)
    else:
        industry_col = []
    if follow_paper:
        valid_time = 1976
        time_period = 12
    else:
        valid_time = 1990
        time_period = 8
    for i in range(len(model)):
        print('Start getting DM NO.' + str(i + 1))
        valid_tick = valid_time + i
        test_tick = time_period + valid_tick
        data_train = data[data['year'] < valid_tick].copy().reset_index(drop=True)
        data_valid = data.loc[(data['year'] >= valid_tick) & (data['year'] < test_tick)].copy().reset_index(drop=True)
        data_test = data[data['year'] >= test_tick].copy().reset_index(drop=True)
        valid_year, all_valid_year = get_variable_list(data_valid, 'year')
        test_year, all_test_year = get_variable_list(data_test, 'year')
        data_train = data_train.drop(columns=list_drop)
        y_valid = data_valid['RET']
        mv_valid = data_valid['mve0']
        data_valid = data_valid.drop(columns=list_drop)
        y_test = data_test['RET']
        mv_test = data_test['mve0']
        data_test = data_test.drop(columns=list_drop)

        if standardization:
            ss = StandardScaler()
            mm = MinMaxScaler()
            t0 = time.time()
            data_train = data_std(data_train, ss, mm, indicators, industry_col, int_col)
            data_valid = data_std(data_valid, ss, mm, indicators, industry_col, int_col, train=False)
            data_test = data_std(data_test, ss, mm, indicators, industry_col, int_col, train=False)
            t1 = time.time()
            print('Finished standardization no.' + str(i + 1) + ' and time is {}s'.format(str(t1 - t0)))

        error_on = time.time()
        error_valid = get_model_error(data_valid, y_valid, model[i], valid_year, all_valid_year)
        error_test = get_model_error(data_test, y_test, model[i], test_year, all_test_year)

        print('Finished error calculation no.' + str(i - valid_time + 1) + ' and time is {}s'.format(
            str(time.time() - error_on)))

        test_dm['Training{}'.format(str(i - valid_time + 1))] = pd.Series(error_test)
        valid_dm['Training{}'.format(str(i - valid_time + 1))] = pd.Series(error_valid)

    return valid_dm, test_dm


def plot_feature_importances(df, model):
    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df = df.drop(columns='importance')
    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:25]))), df['importance_normalized'].head(25), align='center', edgecolor='k')
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:25]))))
    ax.set_yticklabels(df['feature'].head(25))
    # Plot labeling
    plt.xlabel('Normalized Importance')
    plt.title(model)
    plt.show()

    df['importance_normalized'] = df['importance_normalized'] * 100
    return df.sort_values('index').reset_index(drop=True).drop(columns='index')


def rename_dataframe(df, name, start_point=0):
    new_col = [col + '_' + name for col in df.columns[start_point:]]
    old_col = [col for col in df.columns[:start_point]]
    df.columns = old_col + new_col
    return df


def get_named_data(data, name):
    name_col = [col for col in data.columns if name in col]
    return data[name_col]


def plot_r2_score(data, name='test'):
    col = [col[str(col).index('_'):] for col in data.columns]
    col = [c.replace('_adj_', '') for c in col]
    model_name = []
    for item in col:
        if item not in model_name:
            model_name.append(item)
    x = np.arange(len(model_name))
    total_width, n = 0.9, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    score_col = [col for col in data.columns if 'score' in col]
    top_col = [col for col in data.columns if 'top' in col]
    bottom_col = [col for col in data.columns if 'bottom' in col]
    plt.bar(x, data[score_col].mean(), width=width, label='Total')
    plt.bar(x + width, data[top_col].mean(), width=width, label='Top', tick_label=model_name)
    plt.bar(x + 2 * width, data[bottom_col].mean(), width=width, label='Bottom')
    '''
    for a, b in zip(x, data[score_col].mean()):
        plt.text(a, b - 0.1, '%.4f' % b, ha='center', va='bottom', fontsize=10)
    
    for a, b in zip(x + width, data[top_col].mean()):
        plt.text(a, b - 0.3, '%.4f' % b, ha='center', va='bottom', fontsize=10)
    
    for a, b in zip(x + 2 * width, data[bottom_col].mean()):
        plt.text(a, b - 0.2, '%.4f' % b, ha='center', va='bottom', fontsize=10)
    '''
    plt.ylabel('R2 score')
    plt.legend()
    plt.title('R2 score in {} sample'.format(name))


def plot_epoch_score(data, data_name='test', name='score', num=5):
    col = [col for col in data.columns if name in col]
    temp = data.copy()
    temp = temp[col]
    col = [col[str(col).index('_'):] for col in temp.columns]
    model_name = [c.replace('_adj_', '') for c in col]
    temp.columns = model_name

    sample_items = random.sample(model_name, num)

    fig, axes = plt.subplots(len(sample_items), 1, figsize=(20, 16))
    axes[0].set_title('out-of-sample R2 in {} data by each epoch'.format(data_name))
    for i, sample_item in enumerate(sample_items):
        axes[i].plot(temp.index, temp[sample_item], label='model={}'.format(sample_item))
        axes[i].set_ylabel('R2')
        axes[i].legend()


def overall_feature_importance(data):
    data = data.set_index('feature')
    col = [col.replace('importance_normalized_', '') for col in data.columns]
    fe_imp = pd.DataFrame(data=data.values, index=data.index, columns=col)
    ss = StandardScaler()
    fe_imp = pd.DataFrame(data=ss.fit_transform(fe_imp), index=fe_imp.index, columns=fe_imp.columns)
    plt.figure(figsize=(8, 24), dpi=200)
    sns.heatmap(data=fe_imp, cmap=plt.get_cmap('Blues'), cbar=False)
    plt.title('Overall Feature Importance')


def get_average_score(data):
    col = [col[str(col).index('_'):] for col in data.columns]
    col = [c.replace('_adj_', '') for c in col]
    model_name = []
    for item in col:
        if item not in model_name:
            model_name.append(item)
    score = pd.DataFrame(index=['Full', 'top', 'bottom'], columns=model_name)
    score_col = [col for col in data.columns if 'score' in col]
    top_col = [col for col in data.columns if 'top' in col]
    bottom_col = [col for col in data.columns if 'bottom' in col]
    score.iloc[0, :] = data[score_col].mean().values
    score.iloc[1, :] = data[top_col].mean().values
    score.iloc[2, :] = data[bottom_col].mean().values

    return score


def overall_dm_comparison(data):
    temp = data.copy()
    col = [col[str(col).index('_'):] for col in temp.columns]
    col = [c.replace('_', '') for c in col]
    temp.columns = col
    model_name = []
    for item in col:
        if item not in model_name:
            model_name.append(item)

    dataframe = pd.DataFrame(index=model_name[:-1], columns=model_name[1:])
    temp_frame = pd.DataFrame()
    temp_frame['0'] = np.zeros(temp.shape[0])

    for i in range(len(model_name) - 1):
        dm1_name = [c for c in range(len(col)) if col[c] == model_name[i]]
        dm1 = temp.iloc[:, dm1_name]
        for j in range(i + 1, len(model_name)):
            dm2_name = [c for c in range(len(col)) if col[c] == model_name[j]]
            dm2 = temp.iloc[:, dm2_name]
            df = dm_comparison(dm1, dm2, model_name[i], model_name[j])
            temp_frame = pd.concat([temp_frame, df], join='inner', axis=1)
            for k in range(len(df.iloc[:, 0])):
                if (df.iloc[:, 0][k] >= 1000) or (df.iloc[:, 0][k] <= -1000):
                    df = df.drop(index=k)
            df = df.reset_index(drop=True)
            dataframe.iloc[i, j - 1] = df.iloc[:, 0].mean()
            print('Finished DM comparison on ' + model_name[i] + ' and ' + model_name[j])
            del dm2
            gc.collect()

    temp_frame = temp_frame.drop(columns=['0'])

    return dataframe, temp_frame


def plot_dm(dm, name):
    dm = dm.fillna(999)
    f, ax = plt.subplots()
    ax = sns.heatmap(data=dm, cmap=sns.diverging_palette(10, 220, sep=80, n=7), cbar=False, annot=True, ax=ax,
                     mask=dm == 999, annot_kws={'size': 16})
    label_y = ax.get_yticklabels()
    plt.setp(label_y, horizontalalignment='right', fontsize=16)
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=-1, horizontalalignment='right', fontsize=16)
    plt.title('DM for {} data'.format(name))
    plt.show()


def dm_subplot():
    plt.figure(dpi=800)
    fig, axes = plt.subplots(1, 2)
    sns.heatmap(data=test_dm_score, cmap=sns.diverging_palette(10, 220, sep=80, n=7), cbar=False, annot=True,
                ax=axes[0])
    axes[0].set_title('DM for validation data')
    label_y = axes[0].get_yticklabels()
    plt.setp(label_y, rotation=45, horizontalalignment='right')
    label_x = axes[0].get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')
    sns.heatmap(data=valid_dm_score, cmap=sns.diverging_palette(10, 220, sep=80, n=7), cbar=False, annot=True,
                ax=axes[1])
    axes[1].set_title('DM for test data')
    label_y = axes[1].get_yticklabels()
    plt.setp(label_y, rotation=45, horizontalalignment='right')
    label_x = axes[1].get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    start = time.time()
    # Read data
    traindata = pd.read_pickle('GKX_20201231.pkl')
    marco = pd.read_excel('PredictorData2020.xlsx')
    print('Finished reading data in {}s'.format(str(time.time() - start)))

    traindata['year'] = [int(str(date)[:4]) for date in traindata.DATE]
    traindata['month'] = [int(str(date)[:6]) for date in traindata.DATE]
    traindata = traindata.drop(columns='DATE')

    # missing data check
    # miss_month = missing_data_check(traindata, period='month')
    # miss_year = missing_data_check(traindata, period='year')
    # miss_marco = missing_data_check(marco, period='yyyymm')

    # columns to be removed in the dataset, and industry dummy will be kept
    non_list = ['sic2', 'age']  # , 'realestate', 'sin', 'secured', 'stdacc', 'stdcf', 'divi']
    rolling_monthly = ['maxret', 'retvol', 'dolvol', 'idiovol', 'RET']
    traindata = data_processing(traindata, marco, non_list, follow_paper=False)
    
    # Create lags and rolling window
    lag = create_lag(traindata)
    del lag['RET']
    lag = lag[lag['month'] >= 197701].reset_index(drop=True)
    lag = lag.groupby(['month', 'sic2']).transform(lambda x: x.fillna(method='bfill')).fillna(method='ffill')
    roll = create_rolling(traindata, rolling_monthly)
    roll = roll.drop(columns=rolling_monthly)
    roll = roll[roll['month'] >= 197701].reset_index(drop=True)
    roll = roll.groupby(['month', 'sic2']).transform(lambda x: x.fillna(method='bfill')).fillna(method='ffill')
    del lag['permno']
    
    traindata = traindata[traindata['year'] >= 1977].reset_index(drop=True)
    lag = reduce_mem_usage(lag)
    roll = reduce_mem_usage(roll)
    roll_lag = pd.concat([lag, roll], join='inner', axis=1)
    roll_lag = reduce_mem_usage(roll_lag)
    traindata = reduce_mem_usage(traindata)
    # traindata.to_pickle('data.pkl', protocol=4)
    # roll_lag.to_pickle('data_roll_lag.pkl')
    
    # Create Interaction
    macro_fe = ['d/p', 'ep_marco', 'b/m', 'ntis', 'tbl', 'tms', 'dfy', 'svar']
    interaction, interaction_variable = interaction_feature(traindata, macro_fe)
    interaction.to_pickle('interaction.pkl', protocol=4)
    pickle.dump(interaction_variable, open('interact_fe.bin', 'wb'))
    '''
    interaction_variable = pickle.load(open('interact_fe.bin', 'rb'))
    traindata = pd.read_pickle('data.pkl')
    traindata = reduce_mem_usage(traindata)
    roll_lag = pd.read_pickle('data_roll_lag.pkl')
    roll_lag = reduce_mem_usage(roll_lag)
    '''
    if 'age' in traindata.columns:
        del traindata['age']
    roll_lag_col = [col for col in roll_lag.columns if 'dolvol' not in col]
    roll_lag_col = [col for col in roll_lag_col if 'idiovol' not in col]
    roll_lag = roll_lag[roll_lag_col]
    roll_lag = roll_lag.drop(columns='permno')
    traindata = pd.concat([traindata, roll_lag], axis=1, join='inner')
    traindata = reduce_mem_usage(traindata)
    del roll_lag
    gc.collect()

    # interaction = pd.read_pickle('interaction.pkl')
    # interaction = reduce_mem_usage(interaction)
    traindata = pd.concat([traindata, interaction], axis=1, join='inner')
    traindata = reduce_mem_usage(traindata)
    del interaction
    gc.collect()
    int_col = ['age']
    indicators = ['convind', 'divo', 'rd', 'securedind', 'divi']
    drop_list = ['permno', 'month', 'year', 'RET', 'mve0', 'SHROUT', 'prc']
    column = ['score_ord', 'score_adj', 'top_ord', 'top_adj', 'bottom_ord', 'bottom_adj']
    lgb_params = {
        'boosting_type': 'gbdt',
        'n_estimators': 3000,
        'metric': 'rmse',
        'objective': 'huber',
        'learning_rate': 0.015,
        'max_bin': 100,
        'num_leaves': 2 ** 8 - 1,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.7,
        'boost_from_average': False,
        'verbose': -1,
        'tree_learner': 'voting'
    }
    xgb_params = {
        'learning_rate': 0.015,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'max_leaves': 64,
        'grow_policy': 'lossguide',
        'tree_method': 'hist',
        'min_child_weight': 100,
        'objective': 'reg:pseudohubererror',
        'booster': 'gbtree',
        'eval_metric': 'rmse',
        'alpha': 0.1
    }
    # Models
    # OLS
    fi_linear, valid_linear, test_linear, OLS, linear_model, valid_dm_linear, test_dm_linear = model_training(
        traindata, drop_list, LinearRegression(n_jobs=-1), model_type='linear')
    pickle.dump(OLS, open('{}.bin'.format(str(linear_model)), 'wb'))
    valid_linear = rename_dataframe(valid_linear, 'OLS')
    test_linear = rename_dataframe(test_linear, 'OLS')
    valid_dm_linear = rename_dataframe(valid_dm_linear, 'OLS')
    test_dm_inear = rename_dataframe(test_dm_linear, 'OLS')

    # OLS with Huber Loss
    fi_huber, valid_huber, test_huber, Huber, huber_model, valid_dm_huber, test_dm_huber = model_training(
        traindata, drop_list, HuberRegressor(epsilon=1.15), model_type='linear')
    pickle.dump(Huber, open('{}.bin'.format(str(huber_model)), 'wb'))
    valid_huber = rename_dataframe(valid_huber, 'Huber')
    test_huber = rename_dataframe(test_huber, 'Huber')
    valid_dm_huber = rename_dataframe(valid_dm_huber, 'Huber')
    test_dm_huber = rename_dataframe(test_dm_huber, 'Huber')

    # Elastic Net (with and without Huber Loss)
    fi_Enet, valid_Enet, test_Enet, Enet, enet_model, valid_dm_Enet, test_dm_Enet = model_training(
        traindata, drop_list, ElasticNet(alpha=0.1, selection='random'), model_type='linear')
    pickle.dump(Enet, open('{}.bin'.format(str(enet_model)), 'wb'))
    valid_Enet = rename_dataframe(valid_Enet, 'ENet')
    test_Enet = rename_dataframe(test_Enet, 'ENet')
    valid_dm_Enet = rename_dataframe(valid_dm_Enet, 'ElasticNet')
    test_dm_Enet = rename_dataframe(test_dm_Enet, 'ElasticNet')

    fi_Enet_H, valid_Enet_H, test_Enet_H, Enet_H, enet_h_model, valid_dm_Enet_H, test_dm_Enet_H = model_training(
        traindata, drop_list, SGDRegressor(
            loss='huber', penalty='elasticnet', l1_ratio=0.5, epsilon=0.05, early_stopping=True,
            n_iter_no_change=30, verbose=1, learning_rate='adaptive'), model_type='linear')
    pickle.dump(Enet_H, open('Enet_H.bin', 'wb'))
    valid_Enet_H = rename_dataframe(valid_Enet_H, 'ENet_H')
    test_Enet_H = rename_dataframe(test_Enet_H, 'ENet_H')
    valid_dm_Enet_H = rename_dataframe(valid_dm_Enet_H, 'ENet_H')
    test_dm_Enet_H = rename_dataframe(test_dm_Enet_H, 'ENet_H')

    # PCR (with and without Huber Loss)
    pcr = make_pipeline(PCA(), LinearRegression(n_jobs=-1))
    fi_PCR, valid_PCR, test_PCR, PCR, pcr_model, valid_dm_PCR, test_dm_PCR = model_training(
        traindata, drop_list, pcr, model_type='pca')
    pickle.dump(PCR, open('{}.bin'.format(str(pcr_model)), 'wb'))
    valid_PCR = rename_dataframe(valid_PCR, 'PCR')
    test_PCR = rename_dataframe(test_PCR, 'PCR')
    valid_dm_PCR = rename_dataframe(valid_dm_PCR, 'PCR')
    test_dm_PCR = rename_dataframe(test_dm_PCR, 'PCR')

    huber_pcr = make_pipeline(PCA(), HuberRegressor(epsilon=1.15))
    fi_PCR_H, valid_PCR_H, test_PCR_H, PCR_H, pcr_h_model, valid_dm_PCR_H, test_dm_PCR_H = model_training(
        traindata, drop_list, huber_pcr, model_type='pca')
    pickle.dump(PCR_H, open('{}.bin'.format(str(pcr_h_model)), 'wb'))
    valid_PCR_H = rename_dataframe(valid_PCR_H, 'PCR_H')
    test_PCR_H = rename_dataframe(test_PCR_H, 'PCR_H')
    valid_dm_PCR_H = rename_dataframe(valid_dm_PCR_H, 'PCHR')
    test_dm_PCR_H = rename_dataframe(test_dm_PCR_H, 'PCHR')

    # PLS
    fi_PLS, valid_PLS, test_PLS, PLS, pls_model, valid_dm_PLS, test_dm_PLS = model_training(
        traindata, drop_list, PLSRegression(), model_type='linear')
    pickle.dump(PLS, open('{}.bin'.format(str(pls_model)), 'wb'))
    valid_PLS = rename_dataframe(valid_PLS, 'PLS')
    test_PLS = rename_dataframe(test_PLS, 'PLS')
    valid_dm_PLS = rename_dataframe(valid_dm_PLS, 'PLS')
    test_dm_PLS = rename_dataframe(test_dm_PLS, 'PLS')

    # RandomForest
    fi_RF, valid_RF, test_RF, RF, rf_model, valid_dm_RF, test_dm_RF = model_training(
        traindata, drop_list, RandomForestRegressor(n_estimators=200, n_jobs=11, max_depth=6, max_features=0.33,
                                                    verbose=1, max_leaf_nodes=64, oob_score=True, max_samples=0.8, 
                                                    min_samples_split=100, min_samples_leaf=20), 
        model_type='tree', standardization=False)
    pickle.dump(RF, open('{}.bin'.format(str(rf_model)), 'wb'))
    valid_RF = rename_dataframe(valid_RF, 'RF')
    test_RF = rename_dataframe(test_RF, 'RF')
    valid_dm_RF = rename_dataframe(valid_dm_RF, 'RF')
    test_dm_RF = rename_dataframe(test_dm_RF, 'RF')

    # GradientBoosting
    _, valid_HistGBRT, test_HistGBRT, HistGBRT, HistGBRT_model, valid_dm_HistGBRT, test_dm_HistGBRT = model_training(
        traindata, drop_list, HistGradientBoostingRegressor(learning_rate=0.015, l2_regularization=0.2, verbose=1,
                                                            early_stopping=True, max_leaf_nodes=64,
                                                            max_iter=500, scoring='neg_root_mean_squared_error',
                                                            n_iter_no_change=30, tol=1e-5), model_type='none',
        standardization=False, get_importance=False)
    pickle.dump(HistGBRT, open('{}.bin'.format(str(HistGBRT_model)), 'wb'))
    valid_HistGBRT = rename_dataframe(valid_HistGBRT, 'HistGBT')
    test_HistGBRT = rename_dataframe(test_HistGBRT, 'HistGBT')
    valid_dm_HistGBRT = rename_dataframe(valid_dm_HistGBRT, 'HistGBT')
    test_dm_HistGBRT = rename_dataframe(test_dm_HistGBRT, 'HistGBT')

    fi_GBRT, valid_GBRT, test_GBRT, GBRT, GBRT_model, valid_dm_GBRT, test_dm_GBRT = model_training(
        traindata, drop_list, GradientBoostingRegressor(n_estimators=200, loss='huber', verbose=1, max_features=0.33,
                                                        subsample=0.8, max_depth=2, n_iter_no_change=30,
                                                        min_samples_split=100, min_samples_leaf=20,
                                                        max_leaf_nodes=2 ** 8 - 1, learning_rate=0.015),
        model_type='tree', standardization=False)
    pickle.dump(GBRT, open('{}.bin'.format(str(GBRT_model)), 'wb'))
    valid_GBRT = rename_dataframe(valid_GBRT, 'GBRT')
    test_GBRT = rename_dataframe(test_GBRT, 'GBRT')
    valid_dm_GBRT = rename_dataframe(valid_dm_GBRT, 'GBRT')
    test_dm_GBRT = rename_dataframe(test_dm_GBRT, 'GBRT')

    # XGBoost
    fi_XGB, valid_XGB, test_XGB, XGB, XGB_model, valid_dm_XGB, test_dm_XGB = model_training(
        traindata, drop_list, xgb, model_type='xgb', standardization=False)
    pickle.dump(XGB, open('{}.bin'.format(str(XGB_model)), 'wb'))
    valid_XGB = rename_dataframe(valid_XGB, 'XGB')
    test_XGB = rename_dataframe(test_XGB, 'XGB')
    valid_dm_XGB = rename_dataframe(valid_dm_XGB, 'XGB')
    test_dm_XGB = rename_dataframe(test_dm_XGB, 'XGB')

    # LightGBM
    fi_LGB, valid_LGB, test_LGB, LGB, LGB_model, valid_dm_LGB, test_dm_LGB = model_training(
        traindata, drop_list, lgb, model_type='lgbm', standardization=False)
    pickle.dump(LGB, open('{}.bin'.format(str(LGB_model)), 'wb'))
    valid_LGB = rename_dataframe(valid_LGB, 'LGB')
    test_LGB = rename_dataframe(test_LGB, 'LGB')
    valid_dm_LGB = rename_dataframe(valid_dm_LGB, 'LGB')
    test_dm_LGB = rename_dataframe(test_dm_LGB, 'LGB')

    # Multi-layer Perceptron
    _, valid_NN2, test_NN2, NN2, NN2_model, valid_dm_NN2, test_dm_NN2 = model_training(
        traindata, drop_list, MLPRegressor(hidden_layer_sizes=(64, 32), learning_rate='adaptive', early_stopping=True,
                                           verbose=True, alpha=0.01), model_type='network', get_importance=False)
    pickle.dump(NN2, open('{}.bin'.format(str(NN2_model)), 'wb'))
    valid_NN2 = rename_dataframe(valid_NN2, 'NN2')
    test_NN2 = rename_dataframe(test_NN2, 'NN2')
    valid_dm_NN2 = rename_dataframe(valid_dm_NN2, 'NN2')
    test_dm_NN2 = rename_dataframe(test_dm_NN2, 'NN2')

    _, valid_NN3, test_NN3, NN3, NN3_model, valid_dm_NN3, test_dm_NN3 = model_training(
        traindata, drop_list, MLPRegressor(hidden_layer_sizes=(64, 32, 16), learning_rate='adaptive', verbose=True,
                                           early_stopping=True, alpha=0.01), model_type='network', get_importance=False)
    pickle.dump(NN3, open('{}.bin'.format(str(NN3_model)), 'wb'))
    valid_NN3 = rename_dataframe(valid_NN3, 'NN3')
    test_NN3 = rename_dataframe(test_NN3, 'NN3')
    valid_dm_NN3 = rename_dataframe(valid_dm_NN3, 'NN3')
    test_dm_NN3 = rename_dataframe(test_dm_NN3, 'NN3')

    # Generate feature importance figures
    fi_linear = plot_feature_importances(fi_linear, linear_model)
    fi_linear = rename_dataframe(fi_linear, 'linear', 1)

    fi_huber = plot_feature_importances(fi_huber, huber_model)
    fi_huber = rename_dataframe(fi_huber, 'Huber', 1)

    fi_PCR = plot_feature_importances(fi_PCR, pcr_model)
    fi_PCR = rename_dataframe(fi_PCR, 'PCR', 1)

    fi_PCR_H = plot_feature_importances(fi_PCR_H, pcr_h_model)
    fi_PCR_H = rename_dataframe(fi_PCR_H, 'PCR_H', 1)

    fi_Enet_H = plot_feature_importances(fi_Enet_H, 'Enet_H')
    fi_Enet_H = rename_dataframe(fi_Enet_H, 'Enet_H', 1)

    fi_PLS = plot_feature_importances(fi_PLS, pls_model)
    fi_PLS = rename_dataframe(fi_PLS, 'PLS', 1)

    fi_RF = plot_feature_importances(fi_RF, rf_model)
    fi_RF = rename_dataframe(fi_RF, 'RF', 1)

    fi_GBRT = plot_feature_importances(fi_GBRT, 'GBRT')
    fi_GBRT = rename_dataframe(fi_GBRT, 'GBRT', 1)

    fi_XGB = plot_feature_importances(fi_XGB, 'XGBoost')
    fi_XGB = rename_dataframe(fi_XGB, 'XGBoost', 1)

    fi_LGB = plot_feature_importances(fi_LGB, 'LGBM')
    fi_LGB = rename_dataframe(fi_LGB, 'LGBM', 1)

    fi_NN2 = get_model_importance(traindata, drop_list, NN2)
    fi_NN2 = plot_feature_importances(fi_NN2, 'NN2')
    fi_NN2 = rename_dataframe(fi_NN2, 'NN2', 1)

    fi_NN3 = get_model_importance(traindata, drop_list, NN3)
    fi_NN3 = plot_feature_importances(fi_NN3, '3')
    fi_NN3 = rename_dataframe(fi_NN3, 'NN3', 1)

    fi = pd.merge(fi_linear, fi_huber, on='feature')
    fi = fi.merge(fi_PCR_H, on='feature')
    fi = fi.merge(fi_PLS, on='feature')
    fi = fi.merge(fi_RF, on='feature')
    fi = fi.merge(fi_XGB, on='feature')
    fi = fi.merge(fi_LGB, on='feature')
    fi = fi.merge(fi_NN2, on='feature')
    fi = fi.merge(fi_NN3, on='feature')

    overall_feature_importance(fi)

    # Overall out-of-sample R2 score
    test = pd.concat([test_linear, test_huber, test_Enet, test_Enet_H, test_PCR, test_PCR_H, test_PLS, test_RF,
                      test_GBRT, test_HistGBRT, test_XGB, test_LGB, test_NN2, test_NN3], join='inner', axis=1)
    test_adj = get_named_data(test, 'adj')
    test_ord = get_named_data(test, 'ord')
    plot_epoch_score(test_adj)
    plot_r2_score(test_adj)
    test_score_adj = get_average_score(test_adj)
    plot_r2_score(test_ord)
    test_score_ord = get_average_score(test_ord)

    # Overall R2 score in validation set
    valid = pd.concat([valid_linear, valid_huber, valid_Enet, test_Enet_H, valid_PCR, valid_PCR_H, valid_PLS, valid_RF,
                       valid_GBRT, valid_HistGBRT, valid_XGB, valid_LGB, valid_NN2, valid_NN3], join='inner', axis=1)
    valid_adj = get_named_data(valid, 'adj')
    valid_ord = get_named_data(valid, 'ord')
    plot_r2_score(valid_adj, name='valid')
    valid_score_adj = get_average_score(valid_adj)
    plot_r2_score(valid_ord, name='valid')
    valid_score_ord = get_average_score(valid_ord)

    # DM scores
    test_dm = pd.concat([test_dm_linear, test_dm_huber, test_dm_Enet, test_dm_Enet_H, test_dm_PCR, test_dm_PCR_H,
                         test_dm_PLS, test_dm_RF, test_dm_GBRT, test_dm_HistGBRT, test_dm_XGB, test_dm_LGB,
                         test_dm_NN2, test_dm_NN3], join='inner', axis=1)
    test_dm_score, all_test_dm = overall_dm_comparison(test_dm)

    valid_dm = pd.concat([valid_dm_linear, valid_dm_huber, valid_dm_Enet, valid_dm_Enet_H, valid_dm_PCR, valid_dm_PCR_H,
                          valid_dm_PLS, valid_dm_RF, valid_dm_GBRT, valid_dm_HistGBRT, valid_dm_XGB, valid_dm_LGB,
                          valid_dm_NN2, valid_dm_NN3], join='inner', axis=1)
    valid_dm_score, all_valid_dm = overall_dm_comparison(test_dm)

    plot_dm(valid_dm_score, 'valid')
    plot_dm(test_dm_score, 'test')
    end = time.time()
    print(end - start)
