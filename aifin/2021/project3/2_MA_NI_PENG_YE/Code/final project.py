# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:07:34 2021

@author: yanxu
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows',100)
import lightgbm as lgb
from sklearn import preprocessing, metrics
from sklearn.model_selection import TimeSeriesSplit
import gc

#read csv
train_sales = pd.read_csv("C:\\Users\\Nicole MA\\Desktop\\HKUST\\MAFS6010Z\\Project\\Project3\\sales_train_validation.csv")
calendar = pd.read_csv("C:\\Users\\Nicole MA\\Desktop\\HKUST\\MAFS6010Z\\Project\\Project3\\calendar.csv")
submission = pd.read_csv("C:\\Users\\Nicole MA\\Desktop\\HKUST\\MAFS6010Z\\Project\\Project3\\sample_submission.csv")
sell_prices = pd.read_csv("C:\\Users\\Nicole MA\\Desktop\\HKUST\\MAFS6010Z\\Project\\Project3\\sell_prices.csv")

#function for reducing memory usage
#in order to avoid system hardware calculation problem
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

##date re-shaping
#reduce memory usage of calendar and sell prices
calendar = reduce_mem_usage(calendar)
sell_prices = reduce_mem_usage(sell_prices)
#melt all the data we have into useful reasonable train data
#melt every feature exclude ID info
train_sales= pd.melt(train_sales, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
train_sales = reduce_mem_usage(train_sales)

#seperate the validation and evaluation in submission as the test data we used later
test1_rows = [row for row in submission['id'] if 'validation' in row]
test2_rows = [row for row in submission['id'] if 'evaluation' in row]
test1 = submission[submission['id'].isin(test1_rows)]
test2 = submission[submission['id'].isin(test2_rows)]

#change column names to the next 28 days , strated from d_1913
test1.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931', 
                      'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']
#change column names to the next 28 days , strated from d_1941
test2.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 'd_1950', 'd_1951', 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959', 
                      'd_1960', 'd_1961', 'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968', 'd_1969']


# get the list fof unique product ID
product = train_sales[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
# merge product info with ur test data
test2['id'] = test2['id'].str.replace('_evaluation','_validation')
test1 = test1.merge(product, how = 'left', on = 'id')
test2 = test2.merge(product, how = 'left', on = 'id')
test2['id'] = test2['id'].str.replace('_validation','_evaluation')
    
#melt every feature exclude ID info, same process as we done to the total_sales data
test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
#add an extra col which help us identify the part of data
train_sales['part'] = 'train'
test1['part'] = 'test1'
test2['part'] = 'test2'
#concat all the data we processed before, and del the seperated part    
data = pd.concat([train_sales, test1, test2], axis = 0)
del train_sales, test1, test2
#get only a sample for fst training, nrow is seted as we wanted while we do the first model training
#almost cutting half of the training data(25 million rows)
nrows = 25000000
data = data.loc[nrows:]
#drop those useless calendar features
calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)    
#cut the data without test2 part
data = data[data['part'] != 'test2']        
  
#merge in the calendar data and drop the duplicated columns
data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
data.drop(['d', 'day'], inplace = True, axis = 1)
#merge in the sales_prices data, and check the shape of our data
data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1])) 
#memory saving
gc.collect() 

##data exploring
#fill the NaN data of category features
data.isna().sum()
#list out the NaN columns
nan = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
for feature in nan:
    data[feature].fillna('unknown', inplace = True)       
#transform the category features into encoding features
category = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
for feature in category:
    encoder = preprocessing.LabelEncoder()
    data[feature] = encoder.fit_transform(data[feature])


#rolling demand features
data['lag_t28'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
data['rolling_mean_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
data['rolling_std_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())
data['rolling_mean_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
data['rolling_mean_t60'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(60).mean())
data['rolling_mean_t90'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
data['rolling_std_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())
data['rolling_skew_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).skew())
data['rolling_kurt_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).kurt())        
#price features
data['lag_price_t1'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
data['price_change_t1'] = (data['lag_price_t1'] - data['sell_price']) / (data['lag_price_t1'])
data['rolling_price_max_t365'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
data['price_change_t365'] = (data['rolling_price_max_t365'] - data['sell_price']) / (data['rolling_price_max_t365'])
data['rolling_price_std_t7'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
data['rolling_price_std_t30'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
data.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace = True, axis = 1)    
#time features
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['week'] = data['date'].dt.week
data['day'] = data['date'].dt.day
data['dayofweek'] = data['date'].dt.dayofweek

# define list of features
features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'year', 'month', 'week', 'day', 'dayofweek', \
            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 
            'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'lag_t28', 'rolling_mean_t7', 'rolling_std_t7',\
                'rolling_mean_t30', 'rolling_mean_t60','rolling_mean_t90', 
            'rolling_std_t30', 'price_change_t1', 'price_change_t365', 'rolling_price_std_t7', 'rolling_price_std_t30', \
                'rolling_skew_t30', 'rolling_kurt_t30']


##using k fold validation method for our modeling
#left 28*2 = 56 days for test
#all remain if train data
x = data[data['date'] <= '2016-04-24']
y = x['demand']
test = data[(data['date'] > '2016-04-24')]
del data
gc.collect()

params = {
      'num_leaves': 555,
      'min_child_weight': 0.034,
      'feature_fraction': 0.379,
      'bagging_fraction': 0.418,
      'min_data_in_leaf': 106,
      'objective': 'regression',
      'max_depth': -1,
      'learning_rate': 0.007,
      "boosting_type": "gbdt",
      "bagging_seed": 11,
      "metric": 'rmse',
      "verbosity": -1,
      'reg_alpha': 0.3899,
      'reg_lambda': 0.648,
      'random_state': 666,
    }

n_fold = 3 #3 for less time computing
folds = TimeSeriesSplit(n_splits=n_fold)

splits = folds.split(x, y)
y_preds = np.zeros(test.shape[0])
y_oof = np.zeros(x.shape[0])
feature_importances = pd.DataFrame()
feature_importances['feature'] = features
mean_score = []

for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold:',fold_n+1)
    X_train, X_valid = x[features].iloc[train_index], x[features].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    
    model = lgb.train(params, dtrain, num_boost_round=2500, 
                      valid_sets = [dtrain, dvalid],
                      early_stopping_rounds = 50, verbose_eval=100)
    
    
    feature_importances[f'fold_{fold_n + 1}'] = model.feature_importance()
    y_pred_valid = model.predict(X_valid,num_iteration=model.best_iteration)
    y_oof[valid_index] = y_pred_valid
    val_score = np.sqrt(metrics.mean_squared_error(y_pred_valid, y_valid))
    print(f'val rmse score is {val_score}')
    mean_score.append(val_score)
    y_preds += model.predict(test[features], num_iteration=model.best_iteration)/n_fold
    del X_train, X_valid, y_train, y_valid
    gc.collect()

print('mean rmse score over folds is',np.mean(mean_score))
test['demand'] = y_preds



def predict(test, submission):
    predictions = test[['id', 'date', 'demand']]
    predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()
    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    evaluation_rows = [row for row in submission['id'] if 'evaluation' in row] 
    evaluation = submission[submission['id'].isin(evaluation_rows)]

    validation = submission[['id']].merge(predictions, on = 'id')
    final = pd.concat([validation, evaluation])
    final.to_csv('submission.csv', index = False)

predict(test, submission)

#plot out the feature importance
import matplotlib.pyplot as plt
import seaborn as sns
feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)
feature_importances.to_csv('feature_importances.csv')

plt.figure(figsize=(16, 12))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(20), x='average', y='feature');
plt.title('20 TOP feature importance over {} folds average'.format(folds.n_splits));


from itertools import cycle
pd.set_option('max_columns', 50)
plt.style.use('bmh')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

#Visualizing the data for a single item
train_sales = pd.read_csv("C:\\Users\\yanxu\\Desktop\\6010Z\\project3\\sales_train_validation.csv")
d_cols = [c for c in train_sales.columns if 'd_' in c] # sales data columns

train_sales.loc[train_sales['id'] == 'FOODS_3_587_CA_3_validation'] \
    .set_index('id')[d_cols] \
    .T \
    .plot(figsize=(15, 5),
          title='FOODS_3_090_CA_3 sales by "d" number',
          color=next(color_cycle))
plt.legend('')
plt.show()

#counting num of each type
train_sales.groupby('cat_id').count()['id'] \
    .sort_values() \
    .plot(kind='barh', figsize=(15, 5), title='Count of Items by Category')
plt.show()

#Combined Sales by Type
past_sales = train_sales.set_index('id')[d_cols] \
    .T \
    .merge(calendar.set_index('d')['date'],
           left_index=True,
           right_index=True,
            validate='1:1') \
    .set_index('date')


for i in train_sales['cat_id'].unique():
    items_col = [c for c in past_sales.columns if i in c]
    past_sales[items_col] \
        .sum(axis=1) \
        .plot(figsize=(15, 5),
              alpha=0.8,
              title='Total Sales by Item Type')
plt.legend(train_sales['cat_id'].unique())
plt.show()

#distribution of sales price of each state
sell_prices['Category'] = sell_prices['item_id'].str.split('_', expand=True)[0]
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
i = 0
for cat, d in sell_prices.groupby('Category'):
    ax = d['sell_price'].apply(np.log1p) \
        .plot(kind='hist',
                         bins=20,
                         title=f'Distribution of {cat} prices',
                         ax=axs[i],
                                         color=next(color_cycle))
    ax.set_xlabel('Log(price)')
    i += 1
plt.tight_layout()




