import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer as Imputer
from sklearn.manifold import MDS
import lightgbm as lgb
import os
import warnings
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

    _, idx = np.unique(agg, axis = 1, return_index=True)
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

    _, idx = np.unique(categorical, axis = 1, return_index = True)
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

        df_by_loan = df_by_loan.merge(df[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'left')
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
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
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
    train = train.drop(columns = missing_columns)
    test = test.drop(columns = missing_columns)

    return train, test
    
train = pd.read_csv('./home-credit-default-risk/application_train.csv')
train = convert_types(train)
train = pd.get_dummies(train)

test = pd.read_csv('./home-credit-default-risk/application_test.csv')
test = convert_types(test)
test = pd.get_dummies(test)

#train, test = train.align(test, join = 'inner', axis = 1)

#import pdb
#pdb.set_trace()

# previous application
previous = pd.read_csv('./home-credit-default-risk/previous_application.csv')
previous = convert_types(previous, print_info=True)
previous_agg = agg_numeric(previous, 'SK_ID_CURR', 'previous')
print('Previous aggregation shape: ', previous_agg.shape)
previous_counts = agg_categorical(previous, 'SK_ID_CURR', 'previous')
print('Previous counts shape: ', previous_counts.shape)

train = train.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
train = train.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')

test = test.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
test = test.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del previous, previous_agg, previous_counts
gc.collect()

train, test = remove_missing_columns(train, test)

# monthly cash data
cash = pd.read_csv('./home-credit-default-risk/POS_CASH_balance.csv')
cash = convert_types(cash, print_info=True)
cash_by_client = aggregate_client(cash, group_vars=['SK_ID_PREV', 'SK_ID_CURR'], df_names=['cash', 'client'])

train = train.merge(cash_by_client, on = 'SK_ID_CURR', how = 'left')
test = test.merge(cash_by_client, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del cash, cash_by_client
gc.collect()

train, test = remove_missing_columns(train, test)

# monthly credit data
credit = pd.read_csv('./home-credit-default-risk/credit_card_balance.csv')
credit = convert_types(credit, print_info = True)
credit_by_client = aggregate_client(credit, group_vars = ['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['credit', 'client'])

train = train.merge(credit_by_client, on = 'SK_ID_CURR', how = 'left')
test = test.merge(credit_by_client, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del credit, credit_by_client
gc.collect()

train, test = remove_missing_columns(train, test)

# installment payments
installments = pd.read_csv('./home-credit-default-risk/installments_payments.csv')
installments = convert_types(installments, print_info = True)
installments_by_client = aggregate_client(installments, group_vars = ['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['installments', 'client'])

train = train.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left')
test = test.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del installments, installments_by_client
gc.collect()

train, test = remove_missing_columns(train, test)

# bureau_balance and bureau
bureau_balance = pd.read_csv('./home-credit-default-risk/bureau_balance.csv')
bureau_balance = convert_types(bureau_balance, print_info=True)

bureau = pd.read_csv('./home-credit-default-risk/bureau.csv')
bureau = convert_types(bureau, print_info = True)

# 17 columns
bureau_balance_counts = agg_categorical(bureau_balance, parent_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
# 5 columns
bureau_balance_agg = agg_numeric(bureau_balance, parent_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')

# Merge to include the SK_ID_CURR, 24 columns
bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on = 'SK_ID_BUREAU', how = 'left')

#import pdb
#pdb.set_trace()

# 287 columns
bureau_by_client = aggregate_client(bureau_by_loan, group_vars=['SK_ID_BUREAU', 'SK_ID_CURR'], df_names=['bureau', 'client'])

train = train.merge(bureau_by_client, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_by_client, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del bureau, bureau_balance_counts, bureau_balance_agg, bureau_by_loan, bureau_by_client
gc.collect()

train, test = remove_missing_columns(train, test)


def model(features, test_features, encoding='ohe', n_folds=5):
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']

    labels = features['TARGET']

    train_feature_raw = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_feature_raw = test_features.drop(columns = ['SK_ID_CURR'])
    train_feature_raw, test_feature_raw = train_feature_raw.align(test_feature_raw, join='inner',  axis=1)
    feature_names = list(train_feature_raw.columns)

    pipeline = Pipeline(steps = [('imputer', Imputer(strategy = 'median')),
             ('scaler', MinMaxScaler(feature_range = (0, 1))),
             ('pca', PCA())])

             #('mds', MDS(n_components=256))])

    features = pipeline.fit_transform(train_feature_raw)
    # do not refit the pca when testing
    test_features = pipeline.transform(test_feature_raw)

    dim = 512
    features = features[:, :dim]
    test_features = test_features[:, :dim]

    if encoding == 'ohe':
        #features = pd.get_dummies(features)
        #test_features = pd.get_dummies(test_features)
        #import pdb
        #pdb.set_trace()

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

    #feature_names = list(features.columns)
    #features = np.array(features)
    #test_features = np.array(test_features)

    k_fold = KFold(n_splits = n_folds, shuffle = False, random_state = 50)

    feature_importance_values = np.zeros(len(feature_names))
    test_predictions = np.zeros(test_features.shape[0])
    out_of_fold = np.zeros(features.shape[0])

    valid_scores = []
    train_scores = []

    for train_indices, valid_indices in k_fold.split(features):
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
            class_weight = 'balanced', learning_rate = 0.05, 
            reg_alpha = 0.1, reg_lambda = 0.1, 
            subsample = 0.8, n_jobs = -1, random_state = 50)

        model.fit(train_features, train_labels, eval_metric = 'auc',
            eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
            eval_names = ['valid', 'train'], categorical_feature = cat_indices,
            early_stopping_rounds = 100, verbose = 200)

        best_iteration = model.best_iteration_

        #feature_importance_values += model.feature_importances_ / k_fold.n_splits
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits

        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]

        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})

    #feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    valid_auc = roc_auc_score(labels, out_of_fold)
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    fold_names = list(range(n_folds))
    fold_names.append('overall')

    metrics = pd.DataFrame({'fold': fold_names,
        'train': train_scores,
        'valid': valid_scores})

    return submission, metrics

submission, metrics = model(train, test)
submission.to_csv('submission_manualp3_pca_dim512.csv', index = False)
