import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import os
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

import gc


def agg_numeric(df, parent_var, df_name):
    for col in df:
        if col != parent_var and 'SK_ID' in col:
            df = df.drop(columns=col)

    parent_ids = df[parent_var].copy()
    numeric_df = df.select_dtypes('number').copy()
    numeric_df[parent_var] = parent_ids

    agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum'])
    columns = []

    for var in agg.columns.levels[0]:
        if var != parent_var:
            for stat in agg.columns.levels[1]:
                columns.append('%s_%s_%s' % (df_name, var, stat))
    agg.columns = columns

    _, idx = np.unique(agg, axis=1, return_index=True)
    agg = agg.iloc[:, idx]
    return agg


def agg_categorical(df, parent_var, df_name):
    categorical = pd.get_dummies(df.select_dtypes('category'))
    categorical[parent_var] = df[parent_var]

    categorical = categorical.groupby(parent_var).agg(['sum', 'count', 'mean'])
    column_names = []

    for var in categorical.columns.levels[0]:
        for stat in ['sum', 'count', 'mean']:
            column_names.append('%s_%s_%s' % (df_name, var, stat))

    categorical.columns = column_names

    _, idx = np.unique(categorical, axis=1, return_index=True)
    categorical = categorical.iloc[:, idx]
    return categorical


def aggregate_client(df, group_vars, df_names):
    df_agg = agg_numeric(df, parent_var=group_vars[0], df_name=df_names[0])

    if any(df.dtypes == 'category'):
        df_counts = agg_categorical(df, parent_var=group_vars[0], df_name=df_names[0])
        df_by_loan = df_counts.merge(df_agg, on=group_vars[0], how='outer')

        gc.enable()
        del df_agg, df_counts
        gc.collect()

        df_by_loan = df_by_loan.merge(df[[group_vars[0], group_vars[1]]], on=group_vars[0], how='left')
        df_by_loan = df_by_loan.drop(columns=[group_vars[0]])

        df_by_client = agg_numeric(df_by_loan, parent_var=group_vars[1], df_name=df_names[1])

    else:
        df_by_loan = df_agg.merge(df[[group_vars[0], group_vars[1]]], on=group_vars[0], how='left')

        gc.enable()
        del df_agg
        gc.collect()

        df_by_loan = df_by_loan.drop(columns=[group_vars[0]])

        df_by_client = agg_numeric(df_by_loan, parent_var=group_vars[1], df_name=df_names[1])

    gc.enable()
    del df, df_by_loan
    gc.collect()

    return df_by_client


import sys


def return_size(df):
    return round(sys.getsizeof(df) / 1e9, 2)


def convert_types(df, print_info=False):
    original_memory = df.memory_usage().sum()

    for c in df:
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)

        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')

        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)

        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)

        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)

    new_memory = df.memory_usage().sum()

    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
    return df


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})

    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    return mis_val_table_ren_columns


def remove_missing_columns(train, test, threshold=90):
    train_miss = pd.DataFrame(train.isnull().sum())
    train_miss['percent'] = 100 * train_miss[0] / len(train)

    test_miss = pd.DataFrame(test.isnull().sum())
    test_miss['percent'] = 100 * test_miss[0] / len(test)

    missing_train_columns = list(train_miss.index[train_miss['percent'] > threshold])
    missing_test_columns = list(test_miss.index[test_miss['percent'] > threshold])

    missing_columns = list(set(missing_train_columns + missing_test_columns))
    print('There are %d columns with greater than %d%% missing values.' % (len(missing_columns), threshold))
    train = train.drop(columns=missing_columns)
    test = test.drop(columns=missing_columns)

    return train, test


def remove_cor_columns(train, test, threshold=85):
    corr_matrix = train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print('There are %d columns to remove.' % (len(to_drop)))
    train = train.drop(columns=to_drop)
    test = test.drop(columns=to_drop)
    return train, test


train = pd.read_csv('./home-credit-default-risk/application_train.csv')
train = convert_types(train)
test = pd.read_csv('./home-credit-default-risk/application_test.csv')
test = convert_types(test)

# previous application
previous = pd.read_csv('./home-credit-default-risk/previous_application.csv')
previous = convert_types(previous, print_info=True)
previous_agg = agg_numeric(previous, 'SK_ID_CURR', 'previous')
print('Previous aggregation shape: ', previous_agg.shape)
previous_counts = agg_categorical(previous, 'SK_ID_CURR', 'previous')
print('Previous counts shape: ', previous_counts.shape)

train = train.merge(previous_counts, on='SK_ID_CURR', how='left')
train = train.merge(previous_agg, on='SK_ID_CURR', how='left')

test = test.merge(previous_counts, on='SK_ID_CURR', how='left')
test = test.merge(previous_agg, on='SK_ID_CURR', how='left')

gc.enable()
del previous, previous_agg, previous_counts
gc.collect()

train, test = remove_missing_columns(train, test)

# monthly cash data
cash = pd.read_csv('./home-credit-default-risk/POS_CASH_balance.csv')
cash = convert_types(cash, print_info=True)
cash_by_client = aggregate_client(cash, group_vars=['SK_ID_PREV', 'SK_ID_CURR'], df_names=['cash', 'client'])

train = train.merge(cash_by_client, on='SK_ID_CURR', how='left')
test = test.merge(cash_by_client, on='SK_ID_CURR', how='left')

gc.enable()
del cash, cash_by_client
gc.collect()

train, test = remove_missing_columns(train, test)

# monthly credit data
credit = pd.read_csv('./home-credit-default-risk/credit_card_balance.csv')
credit = convert_types(credit, print_info=True)
credit_by_client = aggregate_client(credit, group_vars=['SK_ID_PREV', 'SK_ID_CURR'], df_names=['credit', 'client'])

train = train.merge(credit_by_client, on='SK_ID_CURR', how='left')
test = test.merge(credit_by_client, on='SK_ID_CURR', how='left')

gc.enable()
del credit, credit_by_client
gc.collect()

train, test = remove_missing_columns(train, test)

# installment payments
installments = pd.read_csv('./home-credit-default-risk/installments_payments.csv')
installments = convert_types(installments, print_info=True)
installments_by_client = aggregate_client(installments, group_vars=['SK_ID_PREV', 'SK_ID_CURR'],
                                          df_names=['installments', 'client'])

train = train.merge(installments_by_client, on='SK_ID_CURR', how='left')
test = test.merge(installments_by_client, on='SK_ID_CURR', how='left')

gc.enable()
del installments, installments_by_client
gc.collect()

# train, test = remove_missing_columns(train, test)

# bureau_balance and bureau
bureau_balance = pd.read_csv('./home-credit-default-risk/bureau_balance.csv')
bureau_balance = convert_types(bureau_balance, print_info=True)

bureau = pd.read_csv('./home-credit-default-risk/bureau.csv')
bureau = convert_types(bureau, print_info=True)

# 17 columns
bureau_balance_counts = agg_categorical(bureau_balance, parent_var='SK_ID_BUREAU', df_name='bureau_balance')
# 5 columns
bureau_balance_agg = agg_numeric(bureau_balance, parent_var='SK_ID_BUREAU', df_name='bureau_balance')
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index=True, left_on='SK_ID_BUREAU', how='outer')

# Merge to include the SK_ID_CURR, 24 columns
bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on='SK_ID_BUREAU', how='left')

# import pdb
# pdb.set_trace()

# 287 columns
bureau_by_client = aggregate_client(bureau_by_loan, group_vars=['SK_ID_BUREAU', 'SK_ID_CURR'],
                                    df_names=['bureau', 'client'])

train = train.merge(bureau_by_client, on='SK_ID_CURR', how='left')
test = test.merge(bureau_by_client, on='SK_ID_CURR', how='left')

gc.enable()
del bureau, bureau_balance_counts, bureau_balance_agg, bureau_by_loan, bureau_by_client
gc.collect()

train, test = remove_missing_columns(train, test, threshold=75)


def model(features, test_features, encoding='ohe', n_folds=5):
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']

    labels = features['TARGET']

    features = features.drop(columns=['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns=['SK_ID_CURR'])

    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)

        features, test_features = features.align(test_features, join='inner', axis=1)
        cat_indices = 'auto'
    elif encoding == 'le':
        label_encoder = LabelEncoder()
        cat_indices = []

        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))
                cat_indices.append(i)
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    feature_names = list(features.columns)
    features = np.array(features)
    test_features = np.array(test_features)

    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    feature_importance_values = np.zeros(len(feature_names))
    test_predictions = np.zeros(test_features.shape[0])
    out_of_fold = np.zeros(features.shape[0])

    valid_scores = []
    train_scores = []

    for train_indices, valid_indices in k_fold.split(features):
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

        model = lgb.LGBMClassifier(n_estimators=10000, objective='binary',
                                   class_weight='balanced', learning_rate=0.05,
                                   reg_alpha=0.1, reg_lambda=0.1,
                                   subsample=0.8, n_jobs=-1, random_state=50)

        model.fit(train_features, train_labels, eval_metric='auc',
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names=['valid', 'train'], categorical_feature=cat_indices,
                  early_stopping_rounds=100, verbose=200)

        best_iteration = model.best_iteration_

        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        test_predictions += model.predict_proba(test_features, num_iteration=best_iteration)[:, 1] / k_fold.n_splits

        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration=best_iteration)[:, 1]

        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})

    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    valid_auc = roc_auc_score(labels, out_of_fold)
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    fold_names = list(range(n_folds))
    fold_names.append('overall')

    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    return submission, feature_importances, metrics


def process_feature(features):
    features['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
    features['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)


def poly_process(train, test):
    poly_features_train = train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
    poly_features_test = test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    poly_target = poly_features_train['TARGET']
    poly_features_train = poly_features_train.drop(columns=['TARGET'])
    impt = SimpleImputer(strategy='median')
    poly_features_train = impt.fit_transform(poly_features_train)
    poly_features_test = impt.fit_transform(poly_features_test)
    poly_transformer = PolynomialFeatures(degree=3)
    poly_transformer.fit(poly_features_train)
    poly_features_train = poly_transformer.transform(poly_features_train)
    poly_features_test = poly_transformer.transform(poly_features_test)
    poly_features_train = pd.DataFrame(poly_features_train,
                                       columns=poly_transformer.get_feature_names([
                                           'EXT_SOURCE_1', 'EXT_SOURCE_2',
                                           'EXT_SOURCE_3', 'DAYS_BIRTH'
                                       ]))
    poly_features_test = pd.DataFrame(poly_features_test,
                                      columns=poly_transformer.get_feature_names([
                                          'EXT_SOURCE_1', 'EXT_SOURCE_2',
                                          'EXT_SOURCE_3', 'DAYS_BIRTH'
                                      ]))
    poly_features_train['SK_ID_CURR'] = train['SK_ID_CURR']
    poly_features_test['SK_ID_CURR'] = test['SK_ID_CURR']

    # 参数 on 来指定用于数据集合并的主键
    data_train_poly = train.merge(poly_features_train, on='SK_ID_CURR', how='left')
    data_test_poly = test.merge(poly_features_test, on='SK_ID_CURR', how='left')
    # data_train_poly, data_test_poly = data_train_poly.align(data_test_poly, join='inner', axis=1)
    return data_train_poly, data_test_poly


def domain_knowledge(X):
    X['annuity_income_percentage'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
    X['car_to_birth_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_BIRTH_y']
    X['car_to_employ_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_EMPLOYED']
    X['credit_to_annuity_ratio'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
    X['credit_to_goods_ratio'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
    X['credit_to_income_ratio'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
    X['days_employed_percentage'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH_y']
    X['income_credit_percentage'] = X['AMT_INCOME_TOTAL'] / X['AMT_CREDIT']
    X['payment_rate'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
    X['phone_to_birth_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_BIRTH_y']
    X['phone_to_employ_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_EMPLOYED']
    X['external_sources_weighted'] = X.EXT_SOURCE_1_y * 2 + X.EXT_SOURCE_2_y * 3 + X.EXT_SOURCE_3_y * 4
    for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
        X['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
            X[['EXT_SOURCE_1_y', 'EXT_SOURCE_2_y', 'EXT_SOURCE_3_y']], axis=1)

    X['short_employment'] = (X['DAYS_EMPLOYED'] < -2000).astype(int)
    X['young_age'] = (X['DAYS_BIRTH'] < -14000).astype(int)
    return X


# train = pd.read_csv('train_domain.csv')
# test = pd.read_csv('test_domain.csv')
# feature_importances = pd.read_csv('feature_importance.csv')
# zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
# a = []
# for column in zero_features:
#     if column in train.columns:
#         a.append(column)
# train = train.drop(columns=a)
# test = test.drop(columns=a)
# train, test = remove_cor_columns(train, test)
train = domain_knowledge(train)
test = domain_knowledge(test)
train, test = poly_process(train, test)
# train.to_csv('train.csv')
# test.to_csv('test.csv')
submission, fi, metrics = model(train, test)
submission.to_csv('submission_manualp3.csv', index=False)
