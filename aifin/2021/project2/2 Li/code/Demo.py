import pandas as pd
import numpy as np
import algorithm
import warnings
warnings.filterwarnings('ignore')

# '1957-01-31'
data=pd.read_csv('Data.csv')



def R2_Calculate(fun=algorithm.OLS_3):
    Year_range = pd.date_range(start='1956-01-31', end='2017-01-31', freq='Y')
    Year_range = Year_range.strftime('%Y-%m-%d')
    result = pd.DataFrame(columns=['all', 'top', 'bottom'], index=np.arange(0, 31))
    for start_year in range(len(Year_range) - 30):
        # print(start_year,':')
        # print('train:',Year_range[0],Year_range[start_year+18])
        # print('validation:', Year_range[start_year+18], Year_range[start_year+29])
        # print('test:', Year_range[start_year+29], Year_range[-1])
        df_train_all = data[(data['DATE'] >= Year_range[0]) & (data['DATE'] <= Year_range[start_year + 18])]
        df_train_top, df_train_bottom = algorithm.Data_Process(df_train_all)
        df_validation_all = data[(data['DATE'] > Year_range[start_year + 18]) & (data['DATE'] <= Year_range[start_year + 29])]
        df_validation_top, df_validation_bottom = algorithm.Data_Process(df_validation_all)
        df_test_all = data[(data['DATE'] > Year_range[start_year + 29]) & (data['DATE'] < Year_range[-1])]
        df_test_top, df_test_bottom = algorithm.Data_Process(df_test_all)
        result.iloc[start_year, 0] = fun(df_train_all,df_validation_all, df_test_all)
        result.iloc[start_year, 1] = fun(df_train_top,df_validation_top, df_test_top)
        result.iloc[start_year, 2] = fun(df_train_bottom,df_validation_bottom, df_test_bottom)
        print(result.iloc[start_year,:])
    print(result)
    print(result.mean())
    return 0
R2_Calculate(fun=algorithm.RF)

