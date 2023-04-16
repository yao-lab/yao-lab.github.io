import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, lasso_path, Lasso, ElasticNet
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import os

parameters = {'axes.labelsize': 25,
          'axes.titlesize': 35}
plt.rcParams.update(parameters)
os.makedirs('results', exist_ok=True)
os.makedirs('pics', exist_ok=True)
mode='lasso'


# murder rape robbery assault burglary larceny auto
# rincpc	econgrow	unemp	citypop	term2	term3	term4	a0_5	a5_9	a10_14	a15_19	a20_24	a25_29	citybla	cityfemh	sta_educ	sta_welf	price
# name	city	statenam	state	censdist	year	sworn termlim	jid	mayor	date_wa	date_my	web
# read csvs
df = pd.read_csv('dataset/crime2.csv')
feature_col = df.columns
feature_col = [i for i in feature_col if "Unnamed" not in i]
target = ['murder', 'rape', 'robbery', 'assault', 'burglary', 'larceny', 'auto']
cate_feature_col = ['name',	'city',	'statenam',	'state','censdist',	'year', 'termlim',	'jid',	'mayor',	'date_wa',	'date_my',	'web', 'elecyear',	'governor',]
# cate_feature_col = ['name',	'city',	'statenam',	'state','censdist',	'year', 'termlim',	'jid',	'mayor',	'date_wa',	'date_my',	'web', 'term2',	'term3',	'term4','elecyear',	'governor',]
num_feature_col = [fname for fname in feature_col if fname not in target and fname not in cate_feature_col ]
new_col = [i for i in feature_col if i in target or i in  num_feature_col]
new_df = df[new_col].dropna().reindex()
for ta  in target:
    new_df[ta]=new_df[ta]/ new_df['citypop']
new_df['sworn'] = new_df['sworn'] / new_df['citypop']
new_df['civil'] = new_df['civil'] / new_df['citypop']
# new_df = new_df.drop(columns=['citypop'])
print(new_df)
num_feature_col = [i for i in num_feature_col if 'citypop' not in i]
print(len(num_feature_col))
scaler = StandardScaler()
X = new_df[num_feature_col].reset_index().values
scaler.fit_transform(X)
pca = PCA(n_components=6)
pca.fit(X)
print(pca.explained_variance_ratio_)
plt.figure()
plt.title("PCA for numerical features", fontsize=25)
plt.plot(range(1, 7), pca.explained_variance_ratio_)
plt.ylabel('Variance Ratio', fontsize=25)
plt.xlabel('n component', fontsize=25)
plt.savefig('pics/pca.png')

kf = KFold(n_splits=10)
r_file = open('results/'+mode+'.csv', 'w')
r_file.write('target,train_mse,test_mse\n')
r_file.flush()
# print(num_feature_col[14])
# ssss
num_feature_col = [i for i in num_feature_col if 'citypop' not in i]
for ta in target:
    train_mse = []
    test_mse = []
    for i, (train_index, test_index) in enumerate(kf.split(new_df)):
        scaler = StandardScaler()
        x_train, y_train = new_df[num_feature_col].iloc[train_index], new_df[ta].iloc[train_index].values
        x_test, y_test = new_df[num_feature_col].iloc[test_index], new_df[ta].iloc[test_index].values
        scaler.fit(x_train)
        scaler.transform(x_train)
        scaler.transform(x_test)
        if mode == "lse":
            model = LinearRegression()
        elif mode == "pca_lse":
            model = LinearRegression()
            pca = PCA(n_components=3)
            pca.fit(x_train)
            pca.transform(x_train)
            pca.transform(x_test)
        elif mode == 'lasso':
            model = Lasso(alpha=0.01)
        # elif model == 'lasso_path':
        #     model =  lasso_path()
        elif mode == 'elastic':
            model = ElasticNet()
        elif mode == 'tree':
            model = DecisionTreeRegressor(max_depth=3)
        model.fit(x_train, y_train)
        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)
        train_mse.append(mean_squared_error(pred_train, y_train))
        test_mse.append(mean_squared_error(pred_test, y_test))
    if mode == 'tree':
        # plt.figure(figsize=(15,15))
        # tree.plot_tree(model)
        # plt.savefig('pics/'+ta+'_tree.png')
        # plt.close()
        plt.figure(figsize=(14,10))
        plt.title(ta+' Importantce', fontsize=25)
        plt.bar(range(len(num_feature_col)), model.feature_importances_)
        plt.xticks(range(len(num_feature_col)), num_feature_col, rotation=45, size = 20)
        plt.yticks(fontproperties = 'Times New Roman', size = 20)
        plt.savefig('pics/'+ta+'_feature_imp.png')
        plt.close()
    print(ta, ' Train MSE : ', np.mean(train_mse))
    print(ta, ' Test MSE : ', np.mean(test_mse))
    r_file.write(ta+','+str(np.mean(train_mse))+','+str(np.mean(test_mse))+'\n')
    r_file.flush()
    if mode=='lasso':
        print(model.coef_)
        print(num_feature_col)
        # sss
        alphas_lasso, coefs_lasso, _ = lasso_path(x_train, y_train,)
        select_ind = np.argsort(-np.abs(coefs_lasso[:,-1]))[:5]
        plt.figure(figsize=(12,8))
        colors = (["b", "r", "g", "c", "k"])
        neg_log_alphas_lasso =  -np.log10(alphas_lasso)
        num_feature_col = np.array(num_feature_col)
        plt.title(ta+' Path', fontsize=25)
        for ik, coef_l in enumerate(coefs_lasso[select_ind]):
            
            l1 = plt.plot(neg_log_alphas_lasso, coef_l, label=num_feature_col[select_ind][ik],linewidth=4)
        plt.xlabel("-Log(alpha)", fontsize=25)
        plt.ylabel("coefficients", fontsize=25)
        plt.xticks(fontproperties = 'Times New Roman', size = 18)
        plt.yticks(fontproperties = 'Times New Roman', size = 18)
        plt.legend(fontsize=15)
        # plt.title("Lasso  Paths")
        
# plt.legend((l1[-1], l2[-1]), ("Lasso", "Elastic-Net"), loc="lower left")
# plt.axis("tight")
        plt.savefig('pics/'+ta+'_lasso_path.png')
        plt.close()

        
r_file.close()