import numpy as np
import pandas as pd
import regex as re
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df_calendar = pd.read_csv('./accuracy/calendar.csv', parse_dates=['date'])
df_sales_train = pd.read_csv('./accuracy//sales_train_validation.csv')
df_sell_prices = pd.read_csv('./accuracy//sell_prices.csv')
df_submissions = pd.read_csv('./accuracy//sample_submission.csv')


def to_category(df):
    for name, col_name in df.items():
        if pd.api.types.is_string_dtype(col_name):
            df[name] = col_name.astype('category').cat.as_ordered()


def apply_to_category(df, target):
    for n, c in df.items():
        if (n in target.columns) and (target[n].dtype.name == 'category'):
            df[n] = c.astype('category').cat.as_ordered()
            df[n].cat.set_categories(target[n].cat.categories, ordered=True, inplace=True)


list_id = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
df_day_sales = df_sales_train.drop(list_id, axis=1)

df_sales_train = df_sales_train.melt(id_vars=list_id, value_vars=df_day_sales.columns, var_name='d', value_name='sales')
df_sales_train.drop(['item_id', 'dept_id', 'cat_id', 'store_id'], axis=1, inplace=True)

col_sub = df_submissions.drop(['id'], axis=1).columns
df_submission = df_submissions.melt(id_vars=['id'], value_vars=col_sub, var_name='d', value_name='sales')

df_submission['d'] = df_submission['d'].str.replace('F', '')
df_submission['d'] = pd.to_numeric(df_submission['d'])
df_submission['d'] = df_submission['d'] + 1913
df_submission = df_submission.applymap(str)
df_submission['d'] = 'd_' + df_submission['d'].astype(str)

df_submission = df_submission.merge(df_calendar, left_on='d', right_on='d', how='left')
df_sales_train = df_sales_train.merge(df_calendar, left_on='d', right_on='d', how='left')
df_sell_prices['id'] = df_sell_prices['item_id'] + '_' + df_sell_prices['store_id']
df_sell_prices.drop(['item_id', 'store_id'], axis=1, inplace=True)
df_sell_prices.rename(columns={'id': 'id_for_price'}, inplace=True)

df_sales_train['id_for_price'] = df_sales_train['id'].str.replace('_validation', '')
df_submission['id_for_price'] = df_submission['id'].str.replace('_evaluation', '')
df_submission['id_for_price'] = df_submission['id_for_price'].str.replace('_validation', '')

df_sales_train = pd.merge(df_sales_train, df_sell_prices, how='left', on=['id_for_price', 'wm_yr_wk'])
df_submission = pd.merge(df_submission, df_sell_prices, how='left', on=['id_for_price', 'wm_yr_wk'])

cols_miss = df_submission.columns[df_submission.isnull().any()].tolist()
del df_sell_prices, df_calendar

for col in cols_miss:
    df_sales_train[col + '_was_missing'] = df_sales_train[col].isnull()
    df_submission[col + '_was_missing'] = df_submission[col].isnull()


dict_event = {'event_name_1': 'None', 'event_type_1': 'None',
              'event_name_2': 'None', 'event_type_2': 'None'}

df_sales_train.fillna(value=dict_event, inplace=True)
df_submission.fillna(value=dict_event, inplace=True)

drop_fields = ['weekday', 'year', 'wday', 'month']
df_sales_train.drop(drop_fields, axis=1, inplace=True)
df_submission.drop(drop_fields, axis=1, inplace=True)
df_sales_train.drop(['d'], axis=1, inplace=True)

df_submission_id = df_submission['id']
df_submission_d = df_submission.pop('d')

for name, col_name in df_sales_train.items():
    if pd.api.types.is_string_dtype(col_name):
        df_sales_train[name] = col_name.astype('category').cat.as_ordered()

for i, col in df_submission.items():
    if (i in df_sales_train.columns) and (df_sales_train[i].dtype.name == 'category'):
        df_submission[i] = col.astype('category').cat.as_ordered()
        df_submission[i].cat.set_categories(df_sales_train[i].cat.categories, ordered=True, inplace=True)

df_sales_train.drop(columns=['state_id'], inplace=True)
col_category = df_sales_train.select_dtypes(include='category').columns

for i in col_category:
    df_sales_train['cat_' + i] = df_sales_train[i].cat.codes
    df_submission['cat_' + i] = df_submission[i].cat.codes

df_sales_train.drop(col_category, axis=1, inplace=True)
df_submission.drop(col_category, axis=1, inplace=True)

attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear']
for col in attr:
    df_sales_train[col] = getattr(df_sales_train['date'].dt, col.lower())
    df_submission[col] = getattr(df_submission['date'].dt, col.lower())
df_sales_train.drop('date', axis=1, inplace=True)
df_submission.drop('date', axis=1, inplace=True)

df_sales_train.dropna(axis=0, how='any', inplace=True)

df_sales_train.drop(['sales'], axis=1, inplace=True)
df_submission.drop(['sales'], axis=1, inplace=True)
df_sales_train.to_pickle('df_sales_train.pkl')
df_submission.to_pickle('df_submissions.pkl')
df_submission_id.to_pickle('df_submission_id.pkl')
df_submission_d.to_pickle('df_submission_d.pkl')
