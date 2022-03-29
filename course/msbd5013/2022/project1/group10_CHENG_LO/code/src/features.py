import pandas as pd
import os
import numpy as np

from functools import reduce
from .aggregates import *

def groupby_agg_and_rename(df, groupby_col, agg_dict):
    agg_df = df.groupby(groupby_col).agg(agg_dict)
    agg_df.columns = ["_".join(c) for c in agg_df.columns]
    agg_df = agg_df.reset_index()
    return agg_df

def one_hot_encoder(df, categorical_columns=None, nan_as_category=True):
    """Create a new column for each categorical value in categorical columns. """
    original_columns = list(df.columns)
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    categorical_columns = [c for c in df.columns if c not in original_columns]
    return df, categorical_columns

def application_features(data_dir, mode='train'):
    df = pd.read_csv(os.path.join(data_dir, f'application_{mode}.csv'))
    df, categorical_var = one_hot_encoder(df)

    return {
        "data": df,
        "categorical_var": categorical_var
    }


def bureau_features(data_dir):
    bureau = pd.read_csv(os.path.join(data_dir, 'bureau.csv'))
    bureau_bal = pd.read_csv(os.path.join(data_dir, 'bureau_balance.csv'))
    df = bureau.merge(bureau_bal, on='SK_ID_BUREAU', how='left')

    # get aggregate features
    # need to aggregate features per time
    df = df.sort_values("DAYS_CREDIT")
    df["DAYS_CREDIT_BINNED"] = pd.cut(
        df["DAYS_CREDIT"], 
        [-np.inf, -720, -360, 0], 
        labels=["<2_YEARS", "WITHIN_1_2_YEARS", "WITHIN_1_YEAR"]
    )

    df, categorical_var = one_hot_encoder(df)

    # # aggregation
    df_aggs = groupby_agg_and_rename(df, "SK_ID_CURR", bureau_agg)
    
    return {
        "data": df_aggs,
        "categorical_var": categorical_var,
        "ordinal_var": None,
        "target_var": None,
        "quantitative_var": list(df_aggs.columns)
    }

def previous_app_features(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'previous_application.csv'))

    df, categorical_var = one_hot_encoder(df)

    # treating anomalies as missing in DAYS variables
    df["DAYS_FIRST_DRAWING"] = df["DAYS_FIRST_DRAWING"].replace(
        {365243.000: np.nan}
    )
    df["DAYS_FIRST_DUE"] = df["DAYS_FIRST_DUE"].replace({365243.000: np.nan})
    df["DAYS_LAST_DUE_1ST_VERSION"] = df["DAYS_LAST_DUE_1ST_VERSION"].replace(
        {365243.000: np.nan}
    )
    df["DAYS_LAST_DUE"] = df["DAYS_LAST_DUE"].replace({365243.000: np.nan})
    df["DAYS_TERMINATION"] = df["DAYS_TERMINATION"].replace(
        {365243.000: np.nan}
    )

    df = groupby_agg_and_rename(df, "SK_ID_CURR", previous_agg)
    return {
        "data": df,
        "categorical_var": categorical_var
    }

def credit_card_features(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'credit_card_balance.csv'))
    df = groupby_agg_and_rename(df, "SK_ID_CURR", credit_card_agg)
    return {
        "data": df
    }

def pos_cash_features(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'POS_CASH_balance.csv'))
    df = groupby_agg_and_rename(df, "SK_ID_CURR", pos_cash_agg)
    return {
        "data": df
    }

def installment_features(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'installments_payments.csv'))
    df = groupby_agg_and_rename(df, "SK_ID_CURR", inst_agg)
    return {
        "data": df
    }


def multitable_merge(data_dir):
    train_df = application_features(data_dir, 'train')
    test_df = application_features(data_dir, 'test')
    bureau = bureau_features(data_dir)
    prev_app = previous_app_features(data_dir)
    credit_card = credit_card_features(data_dir)
    pos = pos_cash_features(data_dir)
    installment = installment_features(data_dir)

    train_df = reduce(lambda df_left,df_right: pd.merge(df_left, df_right,on='SK_ID_CURR',how='left'), 
                  [res['data'] for res in [train_df, bureau, prev_app, credit_card, pos, installment]])
    test_df = reduce(lambda df_left,df_right: pd.merge(df_left, df_right,on='SK_ID_CURR',how='left'), 
                  [res['data'] for res in [test_df, bureau, prev_app, credit_card, pos, installment]])
    
    return {
        "train": train_df,
        "test": test_df
    }
