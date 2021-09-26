import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import trange
from multiprocessing import Pool
import gc

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


def data_preprocessing_train_test(data, drop=False):
    if drop:
        # Handle missing values
        total_miss_value = data.isnull().sum().sort_values(ascending=False)
        miss_value_percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
        missing_data_overview = pd.concat([total_miss_value, miss_value_percent], axis=1, keys=['Total', 'Percent'])
        # Drop columns with more than 40% of missing values
        drop_list = missing_data_overview[missing_data_overview['Percent'] > 0.4].index.to_list()
        # some variables are considered important by our model, do not drop them
        drop_list = [col for col in drop_list if col != 'EXT_SOURCE_1' and col != 'OWN_CAR_AGE']
        # Drop columns that are considered not important
        drop_list = drop_list + ['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'REG_REGION_NOT_WORK_REGION',
                                 'LIVE_CITY_NOT_WORK_CITY', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_OWN_CAR',
                                 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                                 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19',
                                 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR']
        data1 = data.drop(columns=drop_list)

    data1 = data.copy()
    # XNA and nan seem to be the same in the description of ORGANIZATION_TYPE when checking feature importance
    data1['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)
    # value 365243 in DAYS_EMPLOYED seems to be outlier, set it to NAN
    data1['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    # the percentage of the days employed relative to the client's age
    data1['DAYS_EMPLOYED_PERC'] = data1['DAYS_EMPLOYED'] / data1['DAYS_BIRTH']
    # the percentage of the credit amount relative to a client's income
    data1['INCOME_CREDIT_PERC'] = data1['AMT_INCOME_TOTAL'] / data1['AMT_CREDIT']
    # Average income per family
    data1['INCOME_PER_PERSON'] = data1['AMT_INCOME_TOTAL'] / data1['CNT_FAM_MEMBERS']
    # the percentage of the loan annuity relative to a client's income
    data1['ANNUITY_INCOME_PERC'] = data1['AMT_ANNUITY'] / data1['AMT_INCOME_TOTAL']
    # the length of the payment in months (since the annuity is the monthly amount due
    data1['PAYMENT_RATE'] = data1['AMT_ANNUITY'] / data1['AMT_CREDIT']

    # The EXT_SOURCEs are considered more important than other features by LGBM model, so we can derive the
    # statistical description of these variables(mean, min, max, sum)
    data1['EXT_SOURCES_MEAN'] = data1[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    data1['EXT_SOURCES_MAX'] = data1[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
    data1['EXT_SOURCES_MIN'] = data1[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    data1['EXT_SOURCES_PERC'] = data1["EXT_SOURCES_MIN"] / data1["EXT_SOURCES_MAX"]
    data1['EXT_SOURCES_NULL_NUM'] = np.sum(data1[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].isnull(), axis=1)
    gc.collect()
    return data1


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def NA_Impute(data, llist):
    missing_value = data.isnull().sum().sort_values(ascending=False)
    fillna_list = missing_value[missing_value > 0].index.to_list()
    # known_list = missing_value[missing_value == 0].index.to_list()
    new_na_list = [col for col in fillna_list if col not in llist]
    known_list_imputation = ['DAYS_BIRTH', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH']
    # Select some features for imputation instead of all for time saving

    for i in trange(len(new_na_list)):
        # use RandomForest to predict missing values
        on = time.time()
        a = [new_na_list[i]] + known_list_imputation
        temp = data[a]
        temp_known = temp[temp[new_na_list[i]].notnull()].values
        temp_unknown = temp[temp[new_na_list[i]].isnull()].values
        rfr = RandomForestRegressor()
        y = temp_known[:, 0]
        x = temp_known[:, 1:]

        pool = Pool(processes=10)
        imputer = pool.apply_async(rfr.fit, args=(x, y))
        pool.close()
        pool.join()
        predict_value = imputer.get().predict(temp_unknown[:, 1:])
        data.loc[(data[new_na_list[i]].isnull()), new_na_list[i]] = predict_value
        off = time.time()
        print('Imputation finished with NO.' + str(i + 1) + ' with time ' + str(off - on))

    return data


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv('D:/HKUST/MAFS6010Z AI in Fintech/home-credit-default-risk/bureau.csv', nrows=num_rows)
    bb = pd.read_csv('D:/HKUST/MAFS6010Z AI in Fintech/home-credit-default-risk/bureau_balance.csv', nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


# Preprocess previous_applications.csv
def previous_applications(num_rows=None, nan_as_category=True):
    prev = pd.read_csv('D:/HKUST/MAFS6010Z AI in Fintech/home-credit-default-risk/previous_application.csv',
                       nrows=num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('D:/HKUST/MAFS6010Z AI in Fintech/home-credit-default-risk/POS_CASH_balance.csv', nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv('D:/HKUST/MAFS6010Z AI in Fintech/home-credit-default-risk/installments_payments.csv',
                      nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv('D:/HKUST/MAFS6010Z AI in Fintech/home-credit-default-risk/credit_card_balance.csv',
                     nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


def dimension_reduction(feature, x_train, x_test):
    # Threshold for removing correlated variables
    threshold = 0.9
    # Absolute value correlation matrix
    corr_matrix = x_train.corr().abs()
    # Upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    for col in to_drop:
        if col in feature['feature'].tolist()[:350]:
            to_drop.remove(col)
    x_train1 = x_train.drop(columns=to_drop)
    x_test1 = x_test.drop(columns=to_drop)
    return x_train1, x_test1


def model_CV(model, x, y):
    begin = time.time()

    pool = Pool(processes=11)
    cv_model = pool.apply_async(model.fit, args=(x, y))
    # use 10-fold cross-validation for model assessment
    cv_result = pool.apply_async(cross_val_score, (cv_model.get(), x, y,), dict(cv=10, scoring='roc_auc'))

    pool.close()
    pool.join()

    # Only the name of classifier is kept
    print(str(model)[:str(model).index('(')] + ' 10 Fold CV Score: ', np.mean(cv_result.get()))

    # Generate feature importance dataframe
    feature_importance_df = pd.DataFrame()
    feats = [f for f in x.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    feature_importance_df['feature'] = feats
    feature_importance_df['importance'] = cv_model.get().feature_importances_

    finish = time.time()
    print('used time: ' + str(finish - begin))
    gc.collect()

    return cv_model.get(), feature_importance_df


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :30].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    # plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')


def plot_repayment_relationship(data, feature):
    temp = data[feature].value_counts()
    temp_y0 = []
    temp_y1 = []
    for val in temp.index:
        temp_y1.append(np.sum(data["TARGET"][data[feature] == val] == 1))
        temp_y0.append(np.sum(data["TARGET"][data[feature] == val] == 0))
    trace1 = go.Bar(
        x=temp.index,
        y=(temp_y1 / temp.sum()) * 100,
        name='NO'
    )
    trace2 = go.Bar(
        x=temp.index,
        y=(temp_y0 / temp.sum()) * 100,
        name='YES'
    )

    data = [trace1, trace2]
    layout = go.Layout(
        title="Distribution of Name of type of the Suite in terms of loan is repayed or not in %",
        width=1000,
        xaxis=dict(
            title='Name of type of the Suite',
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        yaxis=dict(
            title='Count in %',
            titlefont=dict(
                size=16,
                color='rgb(107, 107, 107)'
            ),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        )
    )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


if __name__ == '__main__':
    start = time.time()
    init_notebook_mode(connected=True)
    # Read data
    traindata = pd.read_csv('D:/HKUST/MAFS6010Z AI in Fintech/home-credit-default-risk/application_train.csv')
    testdata = pd.read_csv('D:/HKUST/MAFS6010Z AI in Fintech/home-credit-default-risk/application_test.csv')

    # Data Description
    print(traindata.shape)
    print(traindata['TARGET'].value_counts())
    print(traindata.info())

    # Exploration in terms of loan is re-payed or not
    plot_repayment_relationship(traindata, 'NAME_INCOME_TYPE')
    plot_repayment_relationship(traindata, 'NAME_FAMILY_STATUS')
    plot_repayment_relationship(traindata, 'OCCUPATION_TYPE')
    plot_repayment_relationship(traindata, 'NAME_EDUCATION_TYPE')
    plot_repayment_relationship(traindata, 'NAME_HOUSING_TYPE')
    plot_repayment_relationship(traindata, 'ORGANIZATION_TYPE')
    plot_repayment_relationship(traindata, 'NAME_TYPE_SUITE')

    # Relationship between TARGET and distribution of Ages
    plt.figure(figsize=(10, 8))

    # KDE plot of loans that were repaid on time
    sns.kdeplot(traindata.loc[traindata['TARGET'] == 0, 'DAYS_BIRTH'] / -365, label='target == 0')

    # KDE plot of loans which were not repaid on time
    sns.kdeplot(traindata.loc[traindata['TARGET'] == 1, 'DAYS_BIRTH'] / -365, label='target == 1')

    # Labeling of plot
    plt.xlabel('Age (years)')
    plt.ylabel('Density')
    plt.title('Distribution of Ages')
    plt.legend(labels=['TARGET==0', 'TARGET==1'], loc='upper right')
    plt.show()

    # Display correlations
    correlations = traindata.corr()['TARGET'].sort_values()
    print('Most Positive Correlations:\n', correlations.tail(15))
    print('\nMost Negative Correlations:\n', correlations.head(15))

    # Plot correlation heatmap
    ext_data = traindata[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    ext_data_corrs = ext_data.corr()
    print(ext_data_corrs)
    plt.figure(figsize=(8, 6))

    # Heatmap of correlations
    sns.heatmap(ext_data_corrs, cmap=plt.cm.RdYlBu_r, vmin=-0.25, annot=True, vmax=0.6)
    plt.title('Correlation Heatmap')

    # Plot the extreme values in DAYS_EMPLOYED
    plt.scatter(traindata['DAYS_EMPLOYED'].index, traindata['DAYS_EMPLOYED'].values)
    plt.title('DAYS_EMPLOYED')

    # Relationship between TARGET and EXT_SOURCE
    plt.figure(figsize=(10, 12))
    for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
        # create a new subplot for each source
        plt.subplot(3, 1, i + 1)

        # plot repaid loans
        sns.kdeplot(traindata.loc[traindata['TARGET'] == 0, source], label='target == 0')
        # plot loans that were not repaid
        sns.kdeplot(traindata.loc[traindata['TARGET'] == 1, source], label='target == 1')

        # Label the plots
        plt.title('Distribution of %s by Target Value' % source)
        plt.xlabel('%s' % source)
        plt.ylabel('Density')
        plt.legend(labels=['TARGET==0', 'TARGET==1'], loc='upper right')

    plt.tight_layout(h_pad=0.05)

    train = data_preprocessing_train_test(traindata)
    # Optional: 1 sub-feature is removed since it is not in testdata but shown in traindata
    train = train[~train['CODE_GENDER'].isin(['XNA'])]
    train_label = train['TARGET']  # Take the labels out

    test = data_preprocessing_train_test(testdata)
    test_ID = test['SK_ID_CURR']
    object_value_list = train.select_dtypes('object').apply(pd.Series.nunique,
                                                            axis=0)  # Check if any column has non-numerical value
    train1 = train.copy()
    test1 = test.copy()
    gc.collect()
    '''
    # Imputation with Randomforest
    train = NA_Impute(train, object_value_list)
    test = NA_Impute(test, object_value_list)
    '''
    # One-Hot Encoding
    # train, _ = one_hot_encoder(train1, nan_as_category=True)
    train1, _ = one_hot_encoder(train1, nan_as_category=True)  # Training Data without imputation
    # test, _ = one_hot_encoder(test1, nan_as_category=True)
    test1, _ = one_hot_encoder(test1, nan_as_category=True)  # Testing Data without imputation
    print('Training Features shape: ', train1.shape)
    print('Testing Features shape: ', test1.shape)
    '''
    test.to_csv('test.csv', index=False)
    test1.to_csv('test1.csv', index=False)
    train.to_csv('training.csv', index=False)
    train1.to_csv('training1.csv', index=False)
    '''
    df = train1.append(test1).reset_index()
    bureau = bureau_and_balance()
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    gc.collect()
    prev = previous_applications()
    df = df.join(prev, how='left', on='SK_ID_CURR')
    gc.collect()
    pos = pos_cash()
    df = df.join(pos, how='left', on='SK_ID_CURR')
    gc.collect()
    ins = installments_payments()
    df = df.join(ins, how='left', on='SK_ID_CURR')
    gc.collect()
    cc = credit_card_balance()
    df = df.join(cc, how='left', on='SK_ID_CURR')
    gc.collect()
    df = df.drop(columns=['SK_ID_CURR'])
    training = df[df['TARGET'].notnull()].reset_index(drop=True)
    testing = df[df['TARGET'].isnull()].reset_index(drop=True)
    training = training.drop(columns=['TARGET', 'index'])
    testing = testing.drop(columns=['TARGET', 'index'])
    gc.collect()
    # Some models
    gnb, __ = model_CV(GaussianNB(), train, train_label)
    Rforest, feature_importance_Rforest = model_CV(RandomForestClassifier(), train, train_label)
    Adaboost, feature_importance_Ada = model_CV(AdaBoostClassifier(), train, train_label)
    GradBoost, feature_importance_Grad = model_CV(GradientBoostingClassifier(), train, train_label)
    Xgboost, feature_importance_XG = model_CV(XGBClassifier(), train, train_label)
    lgbm1, feature_importance_nonpa = model_CV(LGBMClassifier(), training, train_label)
    lgbm, feature_importance = model_CV(
        LGBMClassifier(n_estimators=5000, learning_rate=0.05, num_leaves=20, metric='auc', colsample_bytree=0.3,
                       subsample=0.9, max_depth=5, reg_alpha=5, reg_lambda=4, min_split_gain=0.002, min_child_weight=40,
                       silent=True, verbose=-1, n_jobs=16, scale_pos_weight=2), training, train_label)
    display_importances(feature_importance)
    feature_importance_sorted = feature_importance.sort_values(by="importance", ascending=False).reset_index(drop=True)
    # train_1, test_1 = dimension_reduction(feature_importance_sorted, training, testing)
    result = lgbm.predict_proba(testing)[:, 1]
    submission = pd.DataFrame(data={"SK_ID_CURR": test_ID, "TARGET": result})
    submission.to_csv('sub123.csv', index=False)
    end = time.time()
    print(end - start)
