import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import statsmodels.formula.api as smf

train_data=pd.read_csv('application_train.csv')
train_data.set_index('SK_ID_CURR',inplace=True)

def Plot_Histogram(df):
    df['TARGET'].plot.hist()
    plt.show()
    return 0
# Plot_Histogram(train_data)

def Drop_MissingData(df):
    Miss_Number = df.isna().sum()
    Pct_Miss = 100*Miss_Number / len(df)
    Df_MissValue = pd.concat([Miss_Number, Pct_Miss], axis=1)
    Df_MissValue = Df_MissValue.rename(columns={0: 'Missing Value', 1: 'Percentage of Missing Value in Total Value'})
    Df_MissValue = Df_MissValue.sort_values('Percentage of Missing Value in Total Value', ascending=False).round(2)
    Miss_Columns=Df_MissValue[Df_MissValue.iloc[:,1] != 0]
    Filling_Target=Df_MissValue[(Df_MissValue['Percentage of Missing Value in Total Value']<1)&
    (Df_MissValue['Percentage of Missing Value in Total Value']>0)].index
    for col in Filling_Target:
        t=df[col].value_counts()
        df[col].fillna(value=t.index[0],inplace=True)
    # Df_MissValue.to_csv('Df_MissValue.csv')
    Droped_Df=df.dropna(how='any',axis=1)
    print("Total Columns： " + str(df.shape[1]) + "\n"
        "Miss Date Columns: " + str(Miss_Columns.shape[0]-len(Filling_Target)))
    return Droped_Df
# train_data=Drop_MissingData(train_data)

def Dummy_Variable(df):
    NoNum_Name=df.select_dtypes('object').apply(pd.Series.nunique, axis=0)
    # NoNum_Name.to_csv('No-numerical columns.csv')
    df_copy=df.copy()
    le = LabelEncoder()
    le_count = 0
    df_copy = pd.get_dummies(df_copy)
    return df_copy
# train_data=Dummy_Variable(train_data)

def Process_Data(df):
    df_copy=df.copy()
    df_copy=Drop_MissingData(df_copy)
    df_copy=Dummy_Variable(df_copy)
    return df_copy



train_data=Process_Data(train_data)
test_data=pd.read_csv('application_test.csv')
test_data.set_index('SK_ID_CURR',inplace=True)
test_data=Dummy_Variable(test_data)
Col_Name=test_data.columns&train_data.columns
Idx_Name=test_data.index
test_data=test_data[Col_Name]
X_test=Process_Data(test_data)


X_Train=train_data.drop(columns=['TARGET'], axis=1)[Col_Name].values
Y_Train=train_data['TARGET'].values
clf = RandomForestClassifier(max_depth=8, random_state=0)
clf = clf.fit(X_Train, Y_Train)
Y_predict = clf.predict_proba(X_test)[:, 1]

submit=pd.read_csv('sample_submission.csv')
submit['TARGET']=Y_predict
submit.to_csv('submit3.csv',index=False)
