import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df_calendar = pd.read_csv('./accuracy/calendar.csv', parse_dates=['date'])
df_sales_train = pd.read_csv('./accuracy//sales_train_validation.csv')
df_sell_prices = pd.read_csv('./accuracy//sell_prices.csv')
df_submissions = pd.read_csv('./accuracy//sample_submission.csv')


def demo_item_plot(item_id='HOBBIES_1_001'):
    plt.figure(figsize=(12, 3.5))
    date_range = pd.date_range(start='2011-01-29', end='2016-04-24', freq='D')
    for i in range(10):
        df_sales_temp = df_sales_train.loc[df_sales_train['item_id'] == item_id].iloc[i, 6:]
        df_sales_temp.index = date_range
        plt.plot(df_sales_temp)

    plt.title(item_id)
    plt.legend(df_sales_train['store_id'].unique())
    plt.show()
    return


# demo_item_plot()


def demo_item_rolling_plot(item_id='HOBBIES_1_001', period=90):
    plt.figure(figsize=(12, 3.5))
    date_range = pd.date_range(start='2011-01-29', end='2016-04-24', freq='D')
    for i in range(10):
        df_sales_temp = df_sales_train.loc[df_sales_train['item_id'] == item_id].iloc[i, 6:].rolling(period).mean()
        df_sales_temp.index = date_range
        plt.plot(df_sales_temp)
    plt.title(item_id + ', rolling mean ' + str(period) + 'days')
    plt.legend(df_sales_train['store_id'].unique())
    plt.show()
    return


demo_item_rolling_plot(item_id='HOUSEHOLD_1_001', period=90)

def demo_store_rolling_plot(state_id='CA', period=90):
    store_sales = df_sales_train.loc[df_sales_train['state_id'] == state_id]
    date_range = pd.date_range(start='2011-01-29', end='2016-04-24', freq='D')
    plt.figure(figsize=(15, 4))
    for d in store_sales['dept_id'].unique():
        store_sales_temp = store_sales.loc[store_sales['dept_id'] == d]
        store_sales_temp = store_sales_temp.iloc[:, 6:].sum().rolling(period).mean()
        store_sales_temp.index = date_range
        plt.plot(store_sales_temp)
    plt.title(state_id+' sales by department'+' rolling mean '+str(period)+'days')
    plt.legend(store_sales['dept_id'].unique(), loc=(1.0, 0.5))
    plt.show()


# demo_store_rolling_plot(state_id='WI', period=90)


def demo_dept_rolling_plot(dept_id='HOBBIES_1', period=90):
    dept_sales = df_sales_train.loc[df_sales_train['dept_id'] == dept_id]
    date_range = pd.date_range(start='2011-01-29', end='2016-04-24', freq='D')
    plt.figure(figsize=(15, 4))
    for day in dept_sales['store_id'].unique():
        dept_sales_temp = dept_sales.loc[dept_sales['store_id'] == day]
        dept_sales_temp = dept_sales_temp.iloc[:, 6:].sum().rolling(period).mean()
        dept_sales_temp.index = date_range
        plt.plot(dept_sales_temp)
    plt.title(dept_id+' sales by stores, rolling mean '+str(period)+'days')
    plt.legend(dept_sales['store_id'].unique(), loc=(1.0, 0.29))
    plt.show()


# demo_dept_rolling_plot(dept_id='HOBBIES_1', period=90)


def demo_state_rolling_plot(state_id='CA', period=90):
    dept_sales = df_sales_train.loc[df_sales_train['state_id'] == state_id]
    date_range = pd.date_range(start='2011-01-29', end='2016-04-24', freq='D')
    plt.figure(figsize=(15, 4))
    for day in dept_sales['store_id'].unique():
        dept_sales_temp = dept_sales.loc[dept_sales['store_id'] == day]
        dept_sales_temp = dept_sales_temp.iloc[:, 6:].sum().rolling(period).mean()
        dept_sales_temp.index = date_range
        plt.plot(dept_sales_temp)
    plt.title(state_id+' sales by stores, rolling mean '+str(period)+'days')
    plt.legend(dept_sales['store_id'].unique(), loc=(1.0, 0.29))
    plt.show()


# demo_state_rolling_plot(state_id='CA', period=90)


def demo_cat_rolling_plot(cat_id='HOBBIES', period=90):
    dept_sales = df_sales_train.loc[df_sales_train['cat_id'] == cat_id]
    date_range = pd.date_range(start='2011-01-29', end='2016-04-24', freq='D')
    plt.figure(figsize=(15, 4))
    for day in dept_sales['dept_id'].unique():
        dept_sales_temp = dept_sales.loc[dept_sales['dept_id'] == day]
        dept_sales_temp = dept_sales_temp.iloc[:, 6:].sum().rolling(period).mean()
        dept_sales_temp.index = date_range
        plt.plot(dept_sales_temp)
    plt.title(cat_id+' sales by stores, rolling mean '+str(period)+'days')
    plt.legend(dept_sales['dept_id'].unique(), loc=(1.0, 0.29))
    plt.show()

# demo_cat_rolling_plot(cat_id='HOBBIES', period=90)