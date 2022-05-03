import argparse
import algo_utils
import numpy as np
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--fitter', type=str, default='enet')
parser.add_argument('--scaler', type=str, default='Raw', help='Raw,Standard')
parser.add_argument('--mode', type=str, default='ALL', help='ALL,Top,Bottom')
parser.add_argument('--file_path', type=str, default='./GKX_20201231.csv')
parser.add_argument('--save_path', type=str, default='./demo', help='save path')
###### hyper parameters
###linear
parser.add_argument('--ridge_alpha', type=float, default=0.5)
parser.add_argument('--lasso_alpha',type=float, default=1)
parser.add_argument('--enet_alpha',type=float, default=1)
parser.add_argument('--enet_l1_ratio',type=float, default=0.5)
parser.add_argument('--huber_epsilon',type=float, default=1)
parser.add_argument('--huber_alpha',type=float, default=0.01)
parser.add_argument('--pls_n_component',type=int, default=1)
parser.add_argument('--pcs_n_component',type=int, default=1)
###tree
parser.add_argument('--n_estimators',type=int, default=10)
parser.add_argument('--criterion',type=str, default="mse")
parser.add_argument('--max_depth',type=int, default=3)
parser.add_argument('--learning_rate',type=float, default=0.1)
parser.add_argument('--num_leaves',type=int, default=31)
parser.add_argument('--boosting_type',type=str, default='dart')
###nn
parser.add_argument('--layer_1',type=int, default=100)
parser.add_argument('--layer_2',type=int, default=30)
parser.add_argument('--layer_3',type=int, default=30)
parser.add_argument('--layer_4',type=int, default=30)
parser.add_argument('--layer_5',type=int, default=30)
parser.add_argument('--activation',type=str, default='relu')
parser.add_argument('--nn_alpha',type=float, default=0.001)

args = parser.parse_args()
os.makedirs(args.save_path, exist_ok=True)

def get_top(data):
    arr = []
    def top(x):
        x = x.sort_values(by='mvel1')
        arr.append(x.iloc[-1001:-1, :])
    data.groupby('DATE').apply(top)
    df_top = pd.concat(arr)
    return df_top
   
    
def get_bottom(x):
    arr = []
    def bottom(x):
        x = x.sort_values(by='mvel1')
        arr.append(x.iloc[:1000, :])
    data.groupby('DATE').apply(bottom)
    df_top = pd.concat(arr)
    return df_top
   
data = pd.read_csv(args.file_path)

    
    
## get X_Y
result_file = open(os.path.join(args.save_path, 'record.csv'), 'w')
result_file.write('train_r2,val_r2,test_r2')
result_file.write('\n')
result_file.flush()


def missing_values_table(df):
    missing_value = df.isnull().sum() 
    missing_rate = 100 * df.isnull().sum() / len(df) 
    missing_table = pd.concat([missing_value, missing_rate], axis = 1) 
    missing_table_ren_columns = missing_table.rename(columns = {0:'Missing Values',
                                                               1:'Missing Rate'})
    missing_table_ren_columns = missing_table_ren_columns[
        missing_table_ren_columns.iloc[:,1] != 0].sort_values('Missing Rate',ascending=False).round(1)
    return missing_table_ren_columns

def missing_value_filling(df):
    missing_train=missing_values_table(df)
    missing_columns_low = pd.DataFrame(missing_train[missing_train['Missing Rate'] < 50])
    missing_columns_high = pd.DataFrame(missing_train[missing_train['Missing Rate'] >= 50])
    data=pd.DataFrame(df.interpolate())
    data[missing_columns_low.index].fillna(data.mean(),inplace=True)
    data[missing_columns_high.index].fillna(data.mode(),inplace=True)
    # data.fillna(0,inplace=True)
    data=pd.DataFrame(data)
    return data

    

for i in range(0,31):
    data_train=data[(data.DATE>=19600101)&(data.DATE<19780101+i*10000)]
    data_val=data[(data.DATE>=19780101+i*10000)&(data.DATE<19900101+i*10000)]
    data_test=data[(data.DATE>=19900101+i*10000)&(data.DATE<19910101+i*10000)]
    if args.mode == 'ALL':
        pass
    elif args.mode == 'Top':
        data_train  = get_top(data_train)
        data_val  = get_top(data_val)
        data_test  = get_top(data_test)
    elif args.mode == 'Bottom':
        data_train  = get_bottom(data_train)
        data_val  = get_bottom(data_val)
        data_test  = get_bottom(data_test)
    
    
    ###这里可以加其他的fillna 方法
    # data_train = data_train.fillna(0)
    # data_val = data_val.fillna(0)
    # data_test = data_test.fillna(0)
    data_train = missing_value_filling(data_train)
    data_val = missing_value_filling(data_val)
    data_test = missing_value_filling(data_test)
    X_train, Y_train = data_train.drop(['RET','DATE','prc','SHROUT','mve0'],axis=1).copy(), data_train[["RET"]].copy()
    X_val, Y_val = data_val.drop(['RET','DATE','prc','SHROUT','mve0'],axis=1).copy(), data_val[["RET"]].copy()
    X_test, Y_test = data_test.drop(['RET','DATE','prc','SHROUT','mve0'],axis=1).copy(), data_test[["RET"]].copy()
    # X_train, Y_train = data_train.drop("RET",axis=1).copy(), data_train[["RET"]].copy()
    # X_val, Y_val = data_val.drop("RET",axis=1).copy(), data_val[["RET"]].copy()
    # X_test, Y_test = data_test.drop("RET",axis=1).copy(), data_test[["RET"]].copy()
    
        
    ### model
    if args.fitter == 'ols':
        train_r2, val_r2, test_r2, coef = algo_utils.ols(X_train, Y_train,X_val, Y_val,X_test, Y_test,args.scaler)
        tmp_list =[train_r2, val_r2, test_r2, ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()
    elif args.fitter == 'ols3':
        train_r2, val_r2, test_r2, coef= algo_utils.ols(X_train[['mom12m','bm','mvel1']], Y_train,X_val[['mom12m','bm','mvel1']], Y_val, X_test[['mom12m','bm','mvel1']],Y_test,args.scaler)
        tmp_list =[train_r2, val_r2, test_r2 ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()
    elif args.fitter == 'ridge':
        ridge_alpha = args.ridge_alpha 
        param = {'alpha' :ridge_alpha }
        train_r2, val_r2, test_r2, coef = algo_utils.ridge(X_train, Y_train,X_val, Y_val,X_test, Y_test, param)
        tmp_list =[train_r2, val_r2, test_r2, ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()
    elif args.fitter == 'lasso':
        param = {'alpha' :args.lasso_alpha }
        train_r2, val_r2, test_r2, coef = algo_utils.lasso(X_train, Y_train,X_val, Y_val,X_test, Y_test,param)
        tmp_list =[train_r2, val_r2, test_r2, ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()
    elif args.fitter == 'huber':
        param = {'epsilon' :args.huber_epsilon, 'alpha':args.huber_alpha }
        
        train_r2, val_r2, test_r2, coef = algo_utils.huber(X_train, Y_train,X_val, Y_val,X_test, Y_test,param)
        tmp_list =[train_r2, val_r2, test_r2, ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()
    elif args.fitter == 'enet':
        param = {'alpha' :args.enet_alpha, 'l1_ratio':args.enet_l1_ratio }
        train_r2, val_r2, test_r2, coef = algo_utils.enet(X_train, Y_train,X_val, Y_val,X_test, Y_test,param)
        tmp_list =[train_r2, val_r2, test_r2, ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()
    elif args.fitter == 'pls':
        param = {'n_components' :args.pls_n_component }
        train_r2, val_r2, test_r2, coef = algo_utils.pls(X_train, Y_train,X_val, Y_val,X_test, Y_test,param)
        tmp_list =[train_r2, val_r2, test_r2, ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()
    elif args.fitter == 'pcs':
        param = {'n_components' :args.pcs_n_component }
        train_r2, val_r2, test_r2, coef = algo_utils.pcs(X_train, Y_train,X_val, Y_val,X_test, Y_test, param)
        tmp_list =[train_r2, val_r2, test_r2, ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()
        
    elif args.fitter == 'rf':
        param = {'n_estimators' :args.n_estimators,'criterion' :args.criterion,'max_depth':args.max_depth }
        train_r2, val_r2, test_r2, coef = algo_utils.rf(X_train, Y_train,X_val, Y_val,X_test, Y_test, param)
        tmp_list =[train_r2, val_r2, test_r2, ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()
        
    
    elif args.fitter == 'gbdt':
        param = {'n_estimators' :args.n_estimators,'criterion' :args.criterion,'max_depth':args.max_depth, 'learning_rate' : args.learning_rate }
        train_r2, val_r2, test_r2, coef = algo_utils.gbdt(X_train, Y_train,X_val, Y_val,X_test, Y_test,param)
        tmp_list =[train_r2, val_r2, test_r2, ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        print(coef)
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()

        
    elif args.fitter == 'lgb':
        param = {'n_estimators' :args.n_estimators,'num_leaves' :args.num_leaves,'boosting_type' :args.boosting_type,'max_depth':args.max_depth, 'learning_rate' : args.learning_rate }
        train_r2, val_r2, test_r2, coef = algo_utils.lgbr(X_train, Y_train,X_val, Y_val,X_test, Y_test,param)
        tmp_list =[train_r2, val_r2, test_r2, ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()
        
        
    
    elif args.fitter == 'nn1':
        param = {'hidden_layer_sizes': (args.layer_1,1), 'activation' : args.activation, 'alpha' : args.nn_alpha}
        train_r2, val_r2, test_r2, coef = algo_utils.nn(X_train, Y_train,X_val, Y_val,X_test, Y_test,param)
        tmp_list =[train_r2, val_r2, test_r2, ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()
        
       


    elif args.fitter == 'nn2':
        param = {'hidden_layer_sizes': (args.layer_1,args.layer_2,1), 'activation' : args.activation, 'alpha' : args.nn_alpha}
        train_r2, val_r2, test_r2, coef = algo_utils.nn(X_train, Y_train,X_val, Y_val,X_test, Y_test,param)
        tmp_list =[train_r2, val_r2, test_r2, ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()
        
    elif args.fitter == 'nn3':
        param = {'hidden_layer_sizes': (args.layer_1,args.layer_2,args.layer_3,1), 'activation' : args.activation, 'alpha' : args.nn_alpha}
        train_r2, val_r2, test_r2, coef = algo_utils.nn(X_train, Y_train,X_val, Y_val,X_test, Y_test,param)
        tmp_list =[train_r2, val_r2, test_r2, ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()
        
        
    elif args.fitter == 'nn4':
        param = {'hidden_layer_sizes': (args.layer_1,args.layer_2,args.layer_3,args.layer_4,1), 'activation' : args.activation, 'alpha' : args.nn_alpha}
        train_r2, val_r2, test_r2, coef = algo_utils.nn(X_train, Y_train,X_val, Y_val,X_test, Y_test,param)
        tmp_list =[train_r2, val_r2, test_r2, ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()
        
        
    elif args.fitter == 'nn5':
        param = {'hidden_layer_sizes': (args.layer_1,args.layer_2,args.layer_3,args.layer_4,args.layer_5,1), 'activation' : args.activation, 'alpha' : args.nn_alpha}
        train_r2, val_r2, test_r2, coef = algo_utils.nn(X_train, Y_train,X_val, Y_val,X_test, Y_test,param)
        tmp_list =[train_r2, val_r2, test_r2, ]
        tmp_list = [str(i) for i in tmp_list]
        np.save(os.path.join(args.save_path, 'model_' + str(i) +'.npy'), coef)
        result_file.write(','.join(tmp_list))
        result_file.write('\n')
        result_file.flush()

   


result_file.close()