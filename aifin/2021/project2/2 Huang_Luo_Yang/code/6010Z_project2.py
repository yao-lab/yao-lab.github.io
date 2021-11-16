import pandas as pd
import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import time
import gc
from collections import Counter
import seaborn as sns
import inspect
from numba import jit
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, HuberRegressor, ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score


def MSE(T, ori_list, alg_list):
    """
    使用 MSE 作为误差函数。
    :param T: 在计算 g(e{it}) 时的求和上标T
    :param ori_list: 原始时间序列
    :param alg_list: 要比较的的算法
    :return: 返回 g(e{it})
    """
    ret = pow((alg_list[T - 1] - ori_list[T - 1]), 2)
    return ret


def MAPE(T, ori_list, alg_list):
    """
    使用 MAPE 作为误差函数。
    :param T: 在计算 g(e{it}) 时的求和上标T
    :param ori_list: 原始时间序列
    :param alg_list: 要比较的的算法
    :return: 返回 g(e{it})
    """
    ret = abs(alg_list[T - 1] - ori_list[T - 1]) / ori_list[T - 1]
    return ret


def MAE(T, ori_list, alg_list):
    """
    使用 MAE 作为误差函数。
    :param T: 在计算 g(e{it}) 时的求和上标T
    :param ori_list: 原始时间序列
    :param alg_list: 要比较的的算法
    :return: 返回 g(e{it})
    """
    # ret = 0
    # for t in range(T):
    ret = abs((alg_list[T - 1] - ori_list[T - 1]))
    return ret


def cul_d_t(method, ori_list, alg1_list, alg2_list):
    """
    计算d_t.
    :param method: 待使用的误差函数公式
    :param ori_list: 原始时间序列。
    :param alg1_list: 预测算法一的预测结果。
    :param alg2_list: 预测算法二的预测结果。
    :return:  d_t 列表
    """
    d_t_list = []
    if len(ori_list) != len(alg1_list) or len(ori_list) != len(alg2_list):
        raise Exception('The lengths of the three inputs do not match')
    if len(ori_list) == 0:
        raise Exception('The length of the input list should be more than 0')
    list_len = len(ori_list)
    for t in range(1, list_len + 1):
        temp = method(t, ori_list, alg1_list) - method(t, ori_list, alg2_list)
        d_t_list.append(temp)
    return d_t_list


def cul_overline_d(d_t_list):
    """
    计算 d_t 的加和平均，即 overline_d.
    :param d_t_list: d_t 列表
    :return: overline_d
    """
    return sum(d_t_list) / len(d_t_list)


def autocovariance(Xi, N, Xs):
    autoCov = 0
    T = float(N)
    for i in range(0, T):
        autoCov += ((Xi[i]) - Xs) * (Xi[i] - Xs)
    return (1 / (T)) * autoCov


def cul_widehat_gamma_d_tau_list(d_t_list):
    """
    计算 widehat_gamma_d_tau 列表。
    # :param tau: 大于0，小于时间序列长度的数
    :param d_t_list: d_t 列表
    :return: widehat_gamma_d_tau 列表
    """
    widehat_gamma_d_tau_list = []
    overline_d = cul_overline_d(d_t_list)
    for tau in range(len(d_t_list)):
        temp = 0.0
        for t in range(tau, len(d_t_list)):
            temp += (d_t_list[t] - overline_d) * (d_t_list[t - tau] - overline_d)
        widehat_gamma_d_tau_list.append(temp / len(d_t_list))
    return widehat_gamma_d_tau_list


def cul_DM(d_t_list):
    """
    计算 DM 检验的结果。
    :param tau: 大于0，小于时间序列长度的数
    :param d_t_list: d_t 列表
    :return: DM 检验的结果
    """
    T = len(d_t_list)
    widehat_gamma_d_tau_list = cul_widehat_gamma_d_tau_list(d_t_list)
    temp = widehat_gamma_d_tau_list[0]
    overline_d = cul_overline_d(d_t_list)
    DM = overline_d / math.sqrt((temp) / T)
    return DM


def cul_P(d_t_list):
    """
    计算 DM 检验结果的相伴 P 值。
    :param d_t_list: d_t 列表
    :return: 相伴 P 值
    """
    DM = cul_DM(d_t_list)
    return 2 * norm.cdf(-np.abs(DM))


@jit
def missing_data_check(data, start_year, end_year):
    missing_data = pd.DataFrame()
    missing_data.index = data.columns
    for j in range(start_year, end_year):
        begin = time.time()
        temp = data[data['year'] == j]
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


def data_processing(data, list_to_drop, industry_dum_encode=False):
    industry_dummy = data['sic2']
    permno = data['permno']
    date = data['DATE']
    data = data.drop(columns=list_to_drop)
    print('starting to impute missing values')
    start1 = time.time()
    data = data.groupby(['DATE']).transform(lambda x: x.fillna(x.median()))
    temp = pd.concat([permno, industry_dummy], join="inner", axis=1)
    temp = temp.groupby(['permno']).transform(lambda x: x.fillna(x.median()))
    end1 = time.time()
    print('Finished imputing missing values and time is {}s'.format(str(end1 - start1)))
    data_median = pd.concat([date, temp, data], join="inner", axis=1)
    data_median = data_median.dropna(axis=0)
    data_median['market_value'] = data_median['SHROUT'] * data_median['prc']
    if industry_dum_encode:
        str_num = [str(num) for num in data_median['sic2']]
        data_median['sic2'] = str_num
        data_median, col = one_hot_encoder(data_median)
        # data_median.to_csv('median_dummy.csv', index=False)
        return data_median, col
    # data_median.to_csv('median.csv', index=False)
    return data_median.reset_index(drop=True)


# top and bottom 1000 stocks selected by market value
def top_bottom_split(data, mv, y, flag='bottom'):
    newdata = pd.concat([data, mv, y], join="inner", axis=1)
    newdata = newdata.sort_values('market_value', ascending=False).reset_index(drop=True)
    if flag == 'top':
        data1 = newdata.iloc[:1000]
    else:
        data1 = newdata.iloc[-1000:]
    y_new = data1['RET']
    data1 = data1.drop(columns=['RET', 'market_value'])
    return data1, y_new


def data_std(data, stds, rbs, label_kept, rbs_label, train=True):
    temp = data[label_kept]
    temp_rb = data[rbs_label]
    full_label = label_kept + rbs_label
    data = data.drop(columns=full_label)
    org_col = data.columns
    if train:
        standard = stds.fit_transform(data)
        robust = rbs.fit_transform(temp_rb)
        df_std = pd.DataFrame(data=standard, columns=org_col)
        df_rb = pd.DataFrame(data=robust, columns=rbs_label)
    else:
        standard = stds.transform(data)
        robust = rbs.transform(temp_rb)
        df_std = pd.DataFrame(data=standard, columns=org_col)
        df_rb = pd.DataFrame(data=robust, columns=rbs_label)
    new_data = pd.concat([temp, df_std, df_rb], join="inner", axis=1)
    return new_data


def get_feature_importance(indicator, model, present_year, default_year):
    if indicator == 'linear':
        importance = [float(x) for x in model.coef_]
        temp = pd.DataFrame(
            data={'feature': model.feature_names_in_, 'importance{}'.format(str(
                present_year - default_year + 1)): np.abs(importance)}).sort_values('feature').reset_index(drop=True)
    elif indicator == 'pipeline':
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


def df_create(col1, col2, col3, col4, col5, col6, col7, col8):
    if not col3:
        df = pd.DataFrame(data=[col1, col2],
                          index=[retrieve_name(col1), retrieve_name(col2)])
        df = df.T
        df.columns = col7
        df['percentage'] = (df.iloc[:, 0] - df.iloc[:, 1]) / df.iloc[:, 1]
    else:
        df = pd.DataFrame(data=[col1, col2, col3, col4, col5, col6],
                          index=[retrieve_name(col1), retrieve_name(col2), retrieve_name(col3), retrieve_name(col4),
                                 retrieve_name(col5), retrieve_name(col6)])
        df = df.T
        df.columns = col8
        df['percentage'] = (df.iloc[:, 0] - df.iloc[:, 1]) / df.iloc[:, 1]
        df['percentage_valid'] = (df.iloc[:, 2] - df.iloc[:, 3]) / df.iloc[:, 3]
        df['percentage_test'] = (df.iloc[:, 4] - df.iloc[:, 5]) / df.iloc[:, 5]

    return df


@jit
def model_training(data, list_drop, model, model_type, valid_year=1990, end_year=2010, top_bottom=True,
                   standardization=True):
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
    features = pd.DataFrame()
    col = [x for x in data.columns if x not in list_drop]
    col.sort()
    features['feature'] = col
    for i in range(valid_year, 2010):
        start1 = time.time()
        test_year = i + 8
        # train, valid, test sample splitting
        data_train = data[data['year'] < i].copy().reset_index(drop=True)
        data_valid = data.loc[(data['year'] >= i) & (data['year'] < test_year)].copy().reset_index(drop=True)
        data_test = data[data['year'] >= test_year].copy().reset_index(drop=True)
        y_train = data_train['RET']
        data_train = data_train.drop(columns=list_drop)
        y_valid = data_valid['RET']
        mv_valid = data_valid['market_value']
        data_valid = data_valid.drop(columns=list_drop)
        y_test = data_test['RET']
        mv_test = data_test['market_value']
        data_test = data_test.drop(columns=list_drop)
        ss = StandardScaler()
        rs = RobustScaler()
        if standardization:
            t0 = time.time()
            data_train = data_std(data_train, ss, rs, indicators, robust_label)
            data_valid = data_std(data_valid, ss, rs, indicators, robust_label, train=False)
            data_test = data_std(data_test, ss, rs, indicators, robust_label, train=False)
            t1 = time.time()
            print('Finished standardization no.' + str(i - valid_year + 1) + ' and time is {}s'.format(str(t1 - t0)))
        # model fitting and r2
        tick1 = time.time()
        reg = model.fit(data_train, y_train)
        tick2 = time.time()
        print('Finished model fitting no.' + str(i - valid_year + 1) + ' and time is {}s'.format(str(tick2 - tick1)))
        click = time.time()
        temp_feature = get_feature_importance(model_type, reg, i, valid_year)
        features = features.merge(temp_feature, on='feature')
        final = time.time()
        print(
            'Finished feature selection no.' + str(i - valid_year + 1) + ' and time is {}s'.format(str(final - click)))
        valid_score_ord.append(reg.score(data_valid, y_valid))
        valid_score_adj.append(r2_score(y_valid, np.zeros(len(y_valid))))
        test_score_ord.append(reg.score(data_test, y_test))
        test_score_adj.append(r2_score(y_test, np.zeros(len(y_test))))
        if top_bottom:
            # valid top
            valid_top, y_valid_top = top_bottom_split(data_valid, mv_valid, y_valid, flag='top')
            valid_top_ord.append(reg.score(valid_top, y_valid_top))
            valid_top_adj.append(r2_score(y_valid_top, np.zeros(len(y_valid_top))))
            # valid bottom
            valid_bottom, y_valid_bottom = top_bottom_split(data_valid, mv_valid, y_valid)
            valid_bottom_ord.append(reg.score(valid_bottom, y_valid_bottom))
            valid_bottom_adj.append(r2_score(y_valid_bottom, np.zeros(len(y_valid_bottom))))
            # test top
            test_top, y_test_top = top_bottom_split(data_test, mv_test, y_test, flag='top')
            test_top_ord.append(reg.score(test_top, y_test_top))
            test_top_adj.append(r2_score(y_test_top, np.zeros(len(y_test_top))))
            # test bottom
            test_bottom, y_test_bottom = top_bottom_split(data_test, mv_test, y_test)
            test_bottom_ord.append(reg.score(test_bottom, y_test_bottom))
            test_bottom_adj.append(r2_score(y_test_bottom, np.zeros(len(y_test_bottom))))
        end1 = time.time()
        if model_type == 'pipeline':
            print('Finished training no.' + str(i - valid_year + 1) + ' on ' + str(model.steps[0][1])[
                                                                               :str(model.steps[0][1]).index(
                                                                                   '(')] + ' with ' + str(
                model.steps[1][1])[:str(model.steps[1][1]).index('(')] + ' and time is {}s'.format(str(end1 - start1)))
        else:
            print('Finished training no.' + str(i - valid_year + 1) + ' on ' + str(model)[:str(model).index('(')] +
                  ' and time is {}s'.format(str(end1 - start1)))
        gc.collect()
    if model_type == 'none':
        feature_importance = None
    else:
        feature_importance = pd.DataFrame(
            data={'feature': features['feature'], 'importance': features.iloc[:, 1:].mean(axis=1)})
    valid_score = df_create(valid_score_ord, valid_score_adj, valid_top_ord, valid_top_adj, valid_bottom_ord,
                            valid_bottom_adj, column1, column)
    test_score = df_create(test_score_ord, test_score_adj, test_top_ord, test_top_adj, test_bottom_ord, test_bottom_adj,
                           column1, column)
    return feature_importance, valid_score, test_score


def multiprocess_model_training(data, list_drop, model, processes, vital_feature=False, valid_year=1990, end_year=2010,
                                top_bottom=True, standardization=False):
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
    features = pd.DataFrame()
    col = [x for x in data.columns if x not in list_drop]
    col.sort()
    features['feature'] = col
    for i in range(valid_year, end_year):
        start1 = time.time()
        test_year = i + 8
        # train, valid, test sample splitting
        data_train = data[data['year'] < i].copy().reset_index(drop=True)
        data_valid = data.loc[(data['year'] >= i) & (data['year'] < test_year)].copy().reset_index(drop=True)
        data_test = data[data['year'] >= test_year].copy().reset_index(drop=True)
        y_train = data_train['RET']
        data_train = data_train.drop(columns=list_drop)
        y_valid = data_valid['RET']
        mv_valid = data_valid['market_value']
        data_valid = data_valid.drop(columns=list_drop)
        y_test = data_test['RET']
        mv_test = data_test['market_value']
        data_test = data_test.drop(columns=list_drop)
        ss = StandardScaler()
        rs = RobustScaler()
        if standardization:
            t0 = time.time()
            data_train = data_std(data_train, ss, rs, indicators, robust_label)
            data_valid = data_std(data_valid, ss, rs, indicators, robust_label, train=False)
            data_test = data_std(data_test, ss, rs, indicators, robust_label, train=False)
            t1 = time.time()
            print('Finished standardization no.' + str(i - valid_year + 1) + ' and time is {}s'.format(str(t1 - t0)))
        # model fitting and r2 score
        tick1 = time.time()
        pool = Pool(processes=processes)

        reg = pool.apply_async(model.fit, args=(data_train, y_train))

        pool.close()
        pool.join()

        tick2 = time.time()
        print('Finished model fitting and time is {}s'.format(str(tick2 - tick1)))

        valid_score_ord.append(reg.get().score(data_valid, y_valid))
        valid_score_adj.append(r2_score(y_valid, np.zeros(len(y_valid))))
        test_score_ord.append(reg.get().score(data_test, y_test))
        test_score_adj.append(r2_score(y_test, np.zeros(len(y_test))))

        if top_bottom:
            # valid top
            valid_top, y_valid_top = top_bottom_split(data_valid, mv_valid, y_valid, flag='top')
            valid_top_ord.append(reg.get().score(valid_top, y_valid_top))
            valid_top_adj.append(r2_score(y_valid_top, np.zeros(len(y_valid_top))))
            # valid bottom
            valid_bottom, y_valid_bottom = top_bottom_split(data_valid, mv_valid, y_valid)
            valid_bottom_ord.append(reg.get().score(valid_bottom, y_valid_bottom))
            valid_bottom_adj.append(r2_score(y_valid_bottom, np.zeros(len(y_valid_bottom))))
            # test top
            test_top, y_test_top = top_bottom_split(data_test, mv_test, y_test, flag='top')
            test_top_ord.append(reg.get().score(test_top, y_test_top))
            test_top_adj.append(r2_score(y_test_top, np.zeros(len(y_test_top))))
            # test bottom
            test_bottom, y_test_bottom = top_bottom_split(data_test, mv_test, y_test)
            test_bottom_ord.append(reg.get().score(test_bottom, y_test_bottom))
            test_bottom_adj.append(r2_score(y_test_bottom, np.zeros(len(y_test_bottom))))
        if vital_feature:
            click = time.time()
            temp_feature = pd.DataFrame(
                data={'feature': reg.get().feature_names_in_, 'importance{}'.format(str(
                    i - valid_year + 1)): reg.get().feature_importances_}).sort_values('feature').reset_index(drop=True)
            features = features.merge(temp_feature, on='feature')
            final = time.time()
            print(
                'Finished feature selection no.' + str(i - valid_year + 1) + ' and time is {}s'.format(
                    str(final - click)))
        end1 = time.time()
        print('Finished training no.' + str(i - valid_year + 1) + ' on ' + str(model)[:str(model).index('(')] +
              ' and time is {}s'.format(str(end1 - start1)))
        gc.collect()
    if vital_feature:
        feature_importance = pd.DataFrame(
            data={'feature': features['feature'], 'importance': features.iloc[:, 1:].mean(axis=1)})
    else:
        feature_importance = None
    valid_score = df_create(valid_score_ord, valid_score_adj, valid_top_ord, valid_top_adj, valid_bottom_ord,
                            valid_bottom_adj, column1, column)
    test_score = df_create(test_score_ord, test_score_adj, test_top_ord, test_top_adj, test_bottom_ord, test_bottom_adj,
                           column1, column)
    return feature_importance, valid_score, test_score


def plot_feature_importances(df, model, pipeline=False):
    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:25]))),
            df['importance_normalized'].head(25),
            align='center', edgecolor='k')
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:25]))))
    ax.set_yticklabels(df['feature'].head(25))
    # Plot labeling
    plt.xlabel('Normalized Importance')
    if pipeline:
        plt.title('{}'.format(str(model)[:str(model).index('(')]))
    else:
        plt.title('{}'.format(str(model)[:str(model).index('(')]))
    plt.show()

    return df


if __name__ == '__main__':
    start = time.time()
    '''
    # Read data
    traindata = pd.read_csv('D:/HKUST/MAFS6010Z AI in Fintech/project2/GKX_20201231.csv')
    print('Finished reading data')
    
    year = [int(str(date)[:4]) for date in traindata.DATE]
    traindata['year'] = year
    
    miss = missing_data_check(traindata, 1957, 2021)
    traindata = traindata.loc[(traindata['year'] >= 1977) & (traindata['year'] < 2019)]
    
    # Number of stocks at each month
    Number_stocks_per_month = Counter(traindata['DATE'])
    Number_stocks_per_month= pd.DataFrame(Number_stocks_per_month.items(), columns=['Date', 'Numer_of_Stocks'])
    
    int_column_check = ['permno', 'DATE', 'year', 'convind', 'divi', 'divo', 'ps', 'rd', 'securedind', 'sin', 'nincr',
                        'ms', 'ill', 'zerotrade', 'age', 'sic2']
    non_list = ['mve0', 'sic2', 'realestate', 'stdacc', 'stdcf', 'secured', 'divi', 'sin']
    traindata = data_processing(traindata, non_list)
    del year, int_column_check
    gc.collect()
    
    features = traindata.columns[6:100].tolist()
    fig, ax = plt.subplots()
    fig.set_figheight(80)
    fig.set_figwidth(50)
    dt[features].hist(layout=(28, 6), bins=np.linspace(-1,1,50), ax=ax);
    fig.savefig('fig.png')
    
    # Correlation heatmap
    sns.set()
    fig=plt.figure(figsize = (25,25))
    sns.heatmap(data=traindata[features].corr())
    plt.title('Correlation Heatmap')
    plt.show()
    plt.gcf().clear()
    fig.savefig('fig1.png')
    
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    c = dt[features].corr().abs()
    s = c.unstack()
    Sorted = s.sort_values(kind="quicksort").reset_index()
    Sorted.columns = ['col_1','col_2', 'corr']
    Sorted = Sorted.sort_values(by = ['corr', 'col_1'], ascending = False)
    Sorted = Sorted[Sorted['corr']!=1]
    Sorted = Sorted.iloc[::2].reset_index(drop=True)    
    '''
    traindata = pd.read_csv('median.csv')
    indicators = ['convind', 'divo', 'rd', 'securedind']
    robust_label = ['zerotrade']
    drop_list = ['permno', 'DATE', 'year', 'RET', 'market_value', 'SHROUT', 'prc']
    column = ['score_ord', 'score_adj', 'top_ord', 'top_adj', 'bottom_ord', 'bottom_adj']
    column1 = ['score_ord', 'valid_score_adj']
    # Models
    # OLS
    fi_linear, valid_linear, test_linear = model_training(traindata, drop_list, LinearRegression(n_jobs=10),
                                                          model_type='linear')

    # OLS with Huber Loss
    fi_huber, valid_huber, test_huber = model_training(traindata, drop_list, HuberRegressor(epsilon=1.1),
                                                       model_type='linear')

    # Elastic Net
    fi_Enet, valid_Enet, test_Enet = model_training(traindata, drop_list, ElasticNet(alpha=0.1), model_type='linear')

    # PCR (with and without Huber Loss)
    pcr = make_pipeline(PCA(), LinearRegression(n_jobs=10))
    huber_pcr = make_pipeline(PCA(), HuberRegressor(epsilon=1.1))
    fi_PCR, valid_PCR, test_PCR = model_training(traindata, drop_list, pcr, model_type='pipeline')
    fi_PCR_H, valid_PCR_H, test_PCR_H = model_training(traindata, drop_list, huber_pcr, model_type='pipeline')

    # PLS
    fi_PLS, valid_PLS, test_PLS = model_training(traindata, drop_list, PLSRegression(), model_type='linear')
    # RandomForest
    fi_RF, valid_RF, test_RF = model_training(traindata, drop_list, RandomForestRegressor(
        n_estimators=200, n_jobs=-1, max_depth=6, max_features=20, verbose=1), model_type='tree', standardization=False)

    # GradientBoosting
    fi_GBRT, valid_GBRT, test_GBRT = model_training(traindata, drop_list, HistGradientBoostingRegressor(
        verbose=1), model_type='none')
    fi_GBRT1, valid_GBRT1, test_GBRT1 = multiprocess_model_training(traindata, drop_list, GradientBoostingRegressor(
        n_estimators=150, loss='huber', max_depth=2, verbose=1, max_features=20), processes=12, vital_feature=True)
    # Neural Network
    fi_NN, valid_NN, test_NN = multiprocess_model_training(traindata, drop_list, MLPRegressor(
        hidden_layer_sizes=(32, 32, 8), learning_rate='adaptive', early_stopping=True), processes=12)

    # Generate feature importance figures
    fi_linear = plot_feature_importances(fi_linear, LinearRegression())

    end = time.time()
    print(end - start)
