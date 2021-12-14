'''
This part of the code is for the hyperparameter tuning, model fitting and performance analysis for the LSTM model.
The input data should already be created and saved with the naming convention '(asset id).csv', for example '1.csv' for bitcoin
'''

import pandas as pd
import numpy as np
from numpy import array
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import gc
import matplotlib.pyplot as plt

'''
Define useful function
'''
# Split a multivariate sequence into samples
def split_sequences(sequences_X, sequences_Y, n_steps):
	X, y = list(), list()
	for i in range(len(sequences_X)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix >= len(sequences_X):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences_X[i:end_ix,:], sequences_Y[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def split_test_sequences(sequences_X, n_steps):
	X = list()
	for i in range(len(sequences_X)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix >= len(sequences_X):
			break
		# gather input and output parts of the pattern
		seq_x = sequences_X[i:end_ix,:]
		X.append(seq_x)
	return array(X)

# Define a function to downcast the variables to save memory
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

'''
Hyperparameter tuning

n_steps: [30, 60]
units: [20, 40]
LSTM layers: 2 
dropout: 0.2
batch size: 64
epochs: 50
'''
## Define df to store the statistical measure of the result
result_measure_tuning_col = ['a_id', 'n_steps', 'n_units', 'LSTM_layers', 'R2_train', 'R2_val', 'corr_coef_train', 'corr_coef_val']
result_measure_list_tuning = pd.DataFrame(columns = result_measure_tuning_col)

## Define parameters
a_id_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
n_steps_list = [30, 60]
n_units_list = [20, 40]
LSTM_layers_list = [2]
n_epochs = 40
n_batch_size = 64

## Define the sample sizes
n_train_sample = 43200*3 #43200*2
n_val_sample = 43200 #43200
train_start = 0
validate_start = train_start + n_train_sample 
validate_end = validate_start + n_val_sample

## Define the non_predictor
non_predictor = ['timestamp','Asset_ID','Target']

for a_id in a_id_list:
    gc.collect()
    ## Load data 
    df = pd.read_csv(str(a_id) + '.csv', sep=",") #beware of the index
    reduce_mem_usage(df)
    df = df.dropna()
    
    df_train_X = df.iloc[train_start:validate_start,:].drop(columns=non_predictor).reset_index(drop=True)
    df_train_Y = df['Target'].iloc[train_start:validate_start].reset_index(drop=True)
    df_train_Y = df_train_Y.to_numpy()
    predictor_col = df_train_X.columns
    
    ## Data Normalization
    scaler = MinMaxScaler(feature_range = (0, 1))
    df_train_X = scaler.fit_transform(df_train_X)   
                
    for n_steps in n_steps_list:
        ## Convert the training data to right shape for LSTM input
        X, y = split_sequences(df_train_X, df_train_Y, n_steps)
        n_features = X.shape[2]
        
        for n_units in n_units_list:
           for LSTM_layers in LSTM_layers_list:
                ## Define the LSTM model
                if LSTM_layers == 2:
                    model = Sequential()
                    model.add(LSTM(units=n_units, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)))
                    model.add(Dropout(0.2))
                    model.add(LSTM(units=n_units, activation='tanh'))
                    model.add(Dropout(0.2))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                else:
                    model = Sequential()
                    model.add(LSTM(units=n_units, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)))
                    model.add(Dropout(0.2))
                    model.add(LSTM(units=n_units, activation='tanh', return_sequences=True))
                    model.add(Dropout(0.2))
                    model.add(LSTM(units=n_units, activation='tanh'))
                    model.add(Dropout(0.2))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                
                ## Fit the LSTM model with the training set
                model.fit(X, y, epochs = n_epochs, batch_size = n_batch_size)
                
                ## Calculate the R2 and corrcoef of the training set
                train_Y_prediction = model.predict(X)
                R2_train = r2_score(y, train_Y_prediction)
                corr_coef_train = np.corrcoef(y.reshape(len(y),1), train_Y_prediction, rowvar=False)[0,1]
                
                ## Calculate the R2 and corrcoef of the validation set
                df_val_X = df.iloc[validate_start:validate_end,:].drop(columns=non_predictor).reset_index(drop=True)
                df_val_X = pd.concat((df.iloc[validate_start-n_steps+1:validate_start+1,:].drop(columns=non_predictor), df_val_X), axis=0).reset_index(drop=True)
                df_val_X = scaler.transform(df_val_X)
                X_val = split_test_sequences(df_val_X, n_steps)
                Y_prediction = model.predict(X_val)
                df_val_Y = df['Target'].iloc[validate_start:validate_end].reset_index(drop=True)
                df_val_Y = df_val_Y.to_numpy()
                R2_val = r2_score(df_val_Y, Y_prediction)
                corr_coef_val = np.corrcoef(df_val_Y.reshape(len(df_val_Y),1), Y_prediction, rowvar=False)[0,1]
                
                ## Update the statistical measure table
                result_measure_list_tuning = result_measure_list_tuning.append(
                    {'a_id':a_id, 'n_steps':n_steps, 'n_units':n_units, 'LSTM_layers':LSTM_layers, 'R2_train':R2_train,
                     'R2_val':R2_val, 'corr_coef_train':corr_coef_train, 'corr_coef_val':corr_coef_val},ignore_index=True)
                
        ## Export the statistical measure table
        result_measure_list_tuning.to_csv('result_measure_list_LSTM_uptoaid_'+str(a_id)+'.csv',index=False)
                
## Export the statistical measure table
result_measure_list_tuning.to_csv("result_measure_list_LSTM_tuning.csv",index=False)


'''
Model fitting, prediction and result analysis
'''
## Define df to store the feature importance and statistical measure of the result
result_measure_col = ['a_id', 'n_round', 'n_steps', 'n_units', 'LSTM_layers', 'R2_train', 'R2_test', 'corr_coef_train', 'corr_coef_test']
result_measure_list = pd.DataFrame(columns = result_measure_col)
feat_imp = pd.DataFrame()

## Define parameters
a_id_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
n_steps = 30
n_units = 20
LSTM_layers = 2
n_epochs = 35
n_batch_size = 64

## Define the sample sizes
n_sample = 43200*3

## Define the non_predictor
non_predictor = ['timestamp','Asset_ID','Target']


for a_id in a_id_list:   
    gc.collect()
    ## Load data
    df = pd.read_csv(str(a_id) + '.csv', sep=",") #beware of the index
    reduce_mem_usage(df)
    df = df.dropna()  
    
    n_round = 0
    for n_round in range(16):
       ## Define the sample sizes
        train_start = 0 + n_round*n_sample
        test_start = train_start + n_sample 
        test_end = test_start + n_sample
             
        if test_end >= len(df):
           break
        
        df_train_X = df.iloc[train_start:test_start,:].drop(columns=non_predictor).reset_index(drop=True)
        df_train_Y = df['Target'].iloc[train_start:test_start].reset_index(drop=True)
        df_train_Y = df_train_Y.to_numpy()
        predictor_col = df_train_X.columns
        
        ## Data Normalization
        scaler = MinMaxScaler(feature_range = (0, 1))
        df_train_X = scaler.fit_transform(df_train_X)
        
        ## Convert the training data to right shape for LSTM input
        X, y = split_sequences(df_train_X, df_train_Y, n_steps)
        n_features = X.shape[2]
        del(df_train_X)
        del(df_train_Y)
        
        ## Define the LSTM model
        if LSTM_layers == 2:
            model = Sequential()
            model.add(LSTM(units=n_units, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=n_units, activation='tanh'))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
        else:
            model = Sequential()
            model.add(LSTM(units=n_units, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=n_units, activation='tanh', return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=n_units, activation='tanh'))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
        
        ## Fit the LSTM model with the training set
        model.fit(X, y, epochs = n_epochs, batch_size = n_batch_size)
        
        ## Save the model
        model.save('saved_model/my_model_aid' + str(a_id) + '_' + str(n_round))
        
        ## Calculate the R2 and corrcoef of the training set
        train_Y_prediction = model.predict(X)
        R2_train = r2_score(y, train_Y_prediction)
        corr_coef_train = np.corrcoef(y.reshape(len(y),1), train_Y_prediction, rowvar=False)[0,1]
  
        ## Compute the feature importances
        if n_round in [0,3,6,9,12]:
            R_sq_loss_table = pd.DataFrame(columns=predictor_col)
            R_sq_loss_table.loc[0,predictor_col] = 0 
            i = 0
            for pred in predictor_col:
                # We adjust the y_hat by removing the term corresponding feature x features weight
                pred_temp = X[:,:,i]
                X[:,:,i] = 0
                y_hat_new = model.predict(X)
                X[:,:,i] = pred_temp
                R_sq_new = r2_score(y, y_hat_new)
                #print(R_sq_new)
                # Compute loss in R_sq
                loss_in_R2 = R2_train - R_sq_new
                R_sq_loss_table[pred] = loss_in_R2
                i=i+1    
            feat_imp_temp = R_sq_loss_table.iloc[0]/sum(R_sq_loss_table.iloc[0])
            feat_imp[n_round] = feat_imp_temp        

        del(X)
        del(y)
        
        ## Calculate the R2 and corrcoef of the testing set
        df_test_X = df.iloc[test_start:test_end,:].drop(columns=non_predictor).reset_index(drop=True)
        df_test_X = pd.concat((df.iloc[test_start-n_steps+1:test_start+1,:].drop(columns=non_predictor), df_test_X), axis=0).reset_index(drop=True)
        df_test_X = scaler.transform(df_test_X)
        X_test = split_test_sequences(df_test_X, n_steps)
        Y_prediction = model.predict(X_test)
        df_test_Y = df['Target'].iloc[test_start:test_end].reset_index(drop=True)
        df_test_Y = df_test_Y.to_numpy()
        R2_test = r2_score(df_test_Y, Y_prediction)
        corr_coef_test = np.corrcoef(df_test_Y.reshape(len(df_test_Y),1), Y_prediction, rowvar=False)[0,1]
        
        ## Update the statistical measure table
        result_measure_list = result_measure_list.append(
            {'a_id':a_id, 'n_round':n_round, 'n_steps':n_steps, 'n_units':n_units, 'LSTM_layers':LSTM_layers, 'R2_train':R2_train,
             'R2_test':R2_test, 'corr_coef_train':corr_coef_train, 'corr_coef_test':corr_coef_test},ignore_index=True)       

        if n_round==5:
            feat_imp.to_csv('feat_imp_uptoaid_' + str(a_id) + '_5.csv',index=True)
            result_measure_list.to_csv('result_measure_list_LSTM_uptoaid' +str(a_id)+'_5.csv',index=False)
        if n_round==10:
            feat_imp.to_csv('feat_imp_uptoaid_' + str(a_id) + '_10.csv',index=True)
            result_measure_list.to_csv('result_measure_list_LSTM_uptoaid' +str(a_id)+'_10.csv',index=False)
              
        del(df_test_X)
        del(df_test_Y)
        gc.collect()
    
    ## Export the feature importances
    feat_imp.to_csv('feat_imp_uptoaid_'+str(a_id)+'.csv',index=True)
    
    ## Export the statistical measure table
    result_measure_list.to_csv('result_measure_list_LSTM_uptoaid' +str(a_id)+'_10.csv',index=False)

## Calulate the feature importance for each cryptocurrency
asset_name = ['Bitcoin Cash', 'Binance Coin','Bitcoin','EOS.IO','Ethereum Classic','Ethereum','Litecoin','Monero','TRON','Stellar','Cardano','IOTA','Maker','Dogecoin']
for a_id in range(14):
    feat_imp_raw = pd.read_csv('feat_imp_uptoaid_' + str(a_id) + '.csv', sep=",")
    if a_id == 0:
        feat_imp = pd.DataFrame(feat_imp_raw.iloc[:,0])
        feat_imp.columns = ['Feature Name']
    feat_imp_raw['mean'] = feat_imp_raw.mean(axis=1)
    feat_imp[asset_name[a_id]] =  feat_imp_raw['mean']

feat_imp['Average'] = feat_imp.mean(axis=1)
feat_imp.to_csv('feat_imp.csv',index=False)

## Plot the overall feature importance bar chart
plt.figure(figsize = (13, 10))
ax = plt.subplot()
feat_imp_plot = feat_imp.sort_values('Average', ascending = False).reset_index()  
ax.barh(list(reversed(list(feat_imp_plot.index))), 
    feat_imp_plot['Average'], 
    align = 'center', color = 'black')  
# Set the yticks and labels
ax.set_yticks(list(reversed(list(feat_imp_plot.index[:]))))
ax.set_yticklabels(feat_imp_plot['Feature Name'])
# Plot labeling
plt.xlabel('Normalized Importance'); plt.title('LSTM Feature Importances')
plt.show()

## Plot the feature importance bar chart for each cryptocurrency
n_feat_imp = 35
fig, ax = plt.subplots(7,2,figsize = [35, 60])
ax = ax.flatten()
for a_id in range(14):
    feat_imp_plot = feat_imp.sort_values(asset_name[a_id], ascending = False).reset_index()  
    ax[a_id].barh(list(reversed(list(feat_imp_plot.index[:n_feat_imp]))), 
        feat_imp_plot[asset_name[a_id]].head(n_feat_imp), 
        align = 'center', color = 'darkblue')  
    # Set the yticks and labels
    ax[a_id].set_yticks(list(reversed(list(feat_imp_plot.index[:n_feat_imp]))))
    ax[a_id].set_yticklabels(feat_imp_plot['Feature Name'].head(n_feat_imp))
    # Plot labeling
    ax[a_id].set_xlabel('Normalized Importance'); ax[a_id].set_title('LSTM Feature Importances for ' + asset_name[a_id])
plt.show()