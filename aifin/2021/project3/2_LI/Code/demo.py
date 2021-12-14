import pandas as pd
import _pickle as pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Please run the Data_Process.py first
file_df_sales_train = open('df_sales_train.pkl', 'rb')
file_df_submission = open('df_submissions.pkl', 'rb')
file_df_submission_id = open('df_submission_id.pkl', 'rb')
file_df_submission_d = open('df_submission_d.pkl', 'rb')

df_sales_train = pickle.load(file_df_sales_train)
df_submission = pickle.load(file_df_submission)
df_submission_id = pickle.load(file_df_submission_id)
df_submission_d = pickle.load(file_df_submission_d)

df_sales_train = df_sales_train[df_sales_train['Month'].isin([4, 5])]
df_sales_train = df_sales_train[df_sales_train['Year'] >= 2015]
y_train = df_sales_train.pop('sales')
X_train = df_sales_train

# model = RandomForestRegressor(n_jobs=8)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
prediction = model.predict(df_submission)

df_submission['sales'] = prediction
df_submission['id'] = df_submission_id
df_submission['F'] = df_submission_d

dict_submission = {'id': df_submission_id,
                   'F': df_submission_d,
                   'preds': prediction}
file_submission = pd.DataFrame(data=dict_submission)

file_submission['F'] = file_submission['F'].str.replace('d_', '')
file_submission['F'] = pd.to_numeric(file_submission['F'], errors='coerce')
file_submission['F'] = file_submission['F'] - 1913
file_submission['F'] = 'F' + file_submission['F'].astype(str)

file_submission = file_submission.pivot(index='id', columns='F', values='preds')
file_submission = file_submission.reset_index()
file_submission.to_csv('submission.csv', index=False)