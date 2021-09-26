import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import gc
import re
from sklearn.impute import SimpleImputer as Imputer
'''
file:['installments_payments.csv', 'application_train.csv', 'bureau.csv', 'bureau_balance.csv', 'sample_submission.csv', 
      'previous_application.csv', 'POS_CASH_balance.csv', 'HomeCredit_columns_description.csv', 'application_test.csv', 'credit_card_balance.csv']
      
col_application_train:['SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 
'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON'
, 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
'''

data_path = '/data2/home-credit-default-risk'

installments_payments = pd.DataFrame()
application_train = pd.DataFrame()
bureau = pd.DataFrame()
bureau_balance = pd.DataFrame()
sample_submission = pd.DataFrame()
previous_application = pd.DataFrame()
POS_CASH_balance = pd.DataFrame()
HomeCredit_columns_description = pd.DataFrame()
application_test = pd.DataFrame()
credit_card_balance = pd.DataFrame()
col_application_train = []


def check_data(df):
    return df.isnull().sum()/len(df), df.dtypes


def check_error(df):
    df['DAYS_EMPLOYED_flag'] = (df['DAYS_EMPLOYED'] == 365243)
    df['DAYS_EMPLOYED'].replace({365423: np.nan}, inplace=True)
    # OBS_30_CNT_SOCIAL_CIRCLE 疑似异常
    df['OBS_30_CNT_SOCIAL_CIRCLE'][df.OBS_30_CNT_SOCIAL_CIRCLE > 100] = np.nan
    return df


def poly(df):
    return df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]


def lr(train, test, application_test, labels):
    imputer = Imputer(strategy='median')
    scaler = MinMaxScaler(feature_range=(0, 1))

    imputer.fit(train)
    train = imputer.transform(train)
    test = imputer.transform(test)
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    log_reg = LogisticRegression(C=0.0001)
    log_reg.fit(train, labels)
    log_reg_pred = log_reg.predict_proba(test)[:, 1]

    submit = application_test[['SK_ID_CURR']]
    submit['TARGET'] = log_reg_pred
    submit.to_csv('log_reg_baseline.csv', index=False)
    return submit


def lgbm(train, test, labels, num_folds=5):

    test_ids = test['SK_ID_CURR']

    train = train.drop(columns=['SK_ID_CURR'])
    test = test.drop(columns=['SK_ID_CURR'])

    feature_names = list(train.columns)

    train_scores = []
    valid_scores = []

    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test.shape[0])
    feature_importance_value = np.zeros(len(feature_names))

    k_fold = KFold(n_splits=num_folds, shuffle=True, random_state=50)
    col = list(train)
    for i in col:
        if type(i) == tuple:

            rename = i[0]+i[1]
            print(i, rename)
            train.rename(columns={i: rename}, inplace=True)
    train = train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))  # debug: lightgbm.basic.LightGBMError: Do not support special JSON characters in feature name.
    for n_fold, (train_idx, valid_idx) in enumerate(k_fold.split(train, labels)):

        X_train, y_train = train.iloc[train_idx], labels.iloc[train_idx]
        X_valid, y_valid = train.iloc[valid_idx], labels.iloc[valid_idx]

        clf = lgb.LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.018,
            num_leaves=40,
            colsample_bytree=0.95,
            subsample=0.9,
            max_depth=10,
            reg_alpha=0.04,
            reg_lambda=0.07,
            min_split_gain=0.02,
            min_child_weight=40,
            silent=-1,
            verbose=-1, )
        clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_names=['train', 'valid'], eval_metric='auc',
                verbose=200, early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(X_valid, num_iteration=clf.best_iteration_)[:,1]
        sub_preds += clf.predict_proba(test, num_iteration=clf.best_iteration_)[:,1] / k_fold.n_splits
        feature_importance_value += clf.feature_importances_ / k_fold.n_splits

        valid_auc = clf.best_score_['valid']['auc']
        train_auc = clf.best_score_['train']['auc']

        train_scores.append(train_auc)
        valid_scores.append(valid_auc)

        gc.enable()
        del clf, X_train, X_valid, y_train, y_valid
        gc.collect()

    submission = pd.DataFrame({'SK_ID_CURR':test_ids,'TARGET':sub_preds})
    submission.to_csv('lgbm.csv', index=False)
    return submission


def random_forest(train, test, application_test, labels):
    imputer = Imputer(strategy='median')
    scaler = MinMaxScaler(feature_range=(0, 1))

    imputer.fit(train)
    train = imputer.transform(train)
    test = imputer.transform(test)
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    random_forest = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)
    random_forest.fit(train, labels)
    predictions = random_forest.predict_proba(test)[:, 1]
    submit = application_test[['SK_ID_CURR']]
    submit['TARGET'] = predictions
    submit.to_csv('random_forest_baseline.csv', index=False)
    return submit


if __name__ == '__main__':
    print(os.listdir('{}'.format(data_path)))
    for i in os.listdir('{}'.format(data_path)):
        exec("{} = pd.read_csv(\"{}/{}\")".format(i[:-4], data_path, i))
        exec("col_{} = list({})".format(i[:-4], i[:-4]))

    '''
    check  data
    '''
    # application = pd.concat([application_test, application_test])
    missing_rate_at, type_at = check_data(application_train)
    print('average missing rate', missing_rate_at.values.mean(), 'max missing rate', missing_rate_at.values.max(),'    col missing rate', (missing_rate_at != 0).sum()/len(missing_rate_at))
    # print('\n', missing_rate_at.value_counts())
    print(type_at.value_counts())

    '''
    Encoder
    '''
    label_encoder = LabelEncoder()
    encoder_count = 0
    application_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)  # transfer string to num
    for i in col_application_train:
        if type_at[i] == 'object' and len(application_train[i].unique()) <= 2:
            print(i)
            encoder_count += 1
            # label_encoder = label_encoder.fit(application_train[i])
            label_encoder.fit(application_train[i])
            application_train[i] = label_encoder.transform(application_train[i])
            application_test[i] = label_encoder.transform(application_test[i])
    print('num of labelencoder', encoder_count)
    application_train = pd.get_dummies(application_train)  # one hot
    application_test = pd.get_dummies(application_test)  # one hot
    print('features num of train', application_train.shape[1], '.        features num of test', application_test.shape[1])
    labels = application_train.TARGET
    application_train = application_train[list(application_test)]
    application_train['TARGET'] = labels

    '''
    EXCEPTION
    '''
    application_train = check_error(application_train)
    application_test = check_error(application_test)

    '''
    corr 归一化
    '''
    print(application_train.corr()['TARGET'])

    '''
    bureau & bureau_balance
    '''
    print(bureau.groupby('SK_ID_CURR').__len__())
    bureau1 = bureau.drop(columns=['SK_ID_BUREAU', 'CREDIT_CURRENCY', 'CREDIT_TYPE'])
    bureau1 = pd.get_dummies(bureau1)
    bureau_agg = bureau1.groupby('SK_ID_CURR').agg(['mean', 'sum', 'max', 'min'])
    print(list(bureau_agg))
    bureau_agg['NUM_BUREAU'] = bureau.groupby('SK_ID_CURR').count().CREDIT_ACTIVE  # 信用记录笔数
    # print(bureau_agg.dtypes)

    '''
    previous_application
    '''
    print(previous_application.groupby('SK_ID_CURR').__len__())
    prev_app = previous_application.drop(columns=['SK_ID_PREV', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY', 'RATE_INTEREST_PRIVILEGED', 'NAME_CASH_LOAN_PURPOSE', 'NAME_SELLER_INDUSTRY', 'PRODUCT_COMBINATION'])
    for i in ['DAYS_FIRST_DRAWING','DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE','DAYS_TERMINATION']:
        prev_app[i].replace({365423: np.nan}, inplace=True)
    prev_app = pd.get_dummies(prev_app)
    prev_app_agg = prev_app.groupby('SK_ID_CURR').agg(['mean'])
    prev_app_agg['NUM_PREVIOUS_APPLICATION'] = prev_app.groupby('SK_ID_CURR').count().AMT_ANNUITY

    '''
    other
    '''
    pos_cash_agg = pd.DataFrame({'NUM_POSE_CASH_balance': POS_CASH_balance.groupby('SK_ID_CURR').count().SK_ID_PREV})
    credit_ca_agg = pd.DataFrame({'NUM_credit_card_balance': credit_card_balance.groupby('SK_ID_CURR').count().SK_ID_PREV})
    install_agg = pd.DataFrame({'NUM_installments_payments': installments_payments.groupby('SK_ID_CURR').count().SK_ID_PREV})
    '''
    poly-feature
    '''
    poly_train = poly(application_train)
    poly_target = application_train.TARGET
    poly_test = poly(application_test)
    imputer = Imputer(strategy='median')
    poly_train = imputer.fit_transform(poly_train)
    poly_test = imputer.transform(poly_test)
    poly_transformer = PolynomialFeatures(degree=3)
    poly_transformer.fit(poly_train)
    poly_train = poly_transformer.transform(poly_train)
    poly_test = poly_transformer.transform(poly_test)
    poly_train = pd.DataFrame(poly_train, columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))
    poly_test = pd.DataFrame(poly_test, columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))
    poly_train['TARGET'] = poly_target
    print(len(application_train) == len(poly_train))
    print(len(application_test) == len(poly_test))
    poly_train['SK_ID_CURR'] = application_train['SK_ID_CURR']

    '''
    merge
    '''
    train = application_train.merge(poly_train, on='SK_ID_CURR', how='left')
    poly_test['SK_ID_CURR'] = application_test['SK_ID_CURR']
    test = application_test.merge(poly_test, on='SK_ID_CURR', how='left')
    for i in [bureau_agg, prev_app_agg, pos_cash_agg, credit_ca_agg, install_agg]:
        train = train.merge(i, on='SK_ID_CURR', how='left')
        test = test.merge(i, on='SK_ID_CURR', how='left')
    for i in list(train):
        if i[:4] == 'NUM_' or (type(i) == tuple and i[0][:4] == 'NUM_'):
            print(i)
            rename = i
            if type(i) == tuple:
                rename = i[0] + i[1]
                train.rename(columns={i: rename}, inplace=True)
                test.rename(columns={i: rename}, inplace=True)
            train_t = train[rename]
            train_t.fillna(0, inplace=True)
            train[rename] = train_t
            test_t = test[rename]
            test[rename].fillna(0, inplace=True)
            test[rename] = test_t
    # train, test = train.align(test, join='inner', axis=1)

    '''
    model
    '''

    for i in list(train):
        if 'TARGET' in i:
            print(i)
    train = train.drop(columns=['TARGET_x', 'TARGET_y'])

    #re_lr = lr(train, test, application_test, labels)  # logistic regression
    # re_rf = random_forest(train, test, application_test, labels)   # random forest
    re_lgbm = lgbm(train, test, labels, 5)  # light gbm

    #re_comb = pd.DataFrame({'SK_ID_CURR': test.SK_ID_CURR, 'TARGET': (re_lr.TARGET + re_lgbm.TARGET)/2})
    #re_comb.to_csv('re_comb.csv', index=False)




