import pandas as pd
import numpy as np
import scipy.stats as stats
import heapq
import matplotlib.pyplot as plt
from HodgeRank import hodge_rank

# Read Data
College_data = pd.read_csv('College_Score_Name.csv')['College Score'].values
College_Name = pd.read_csv('College_Score_Name.csv')['College Name'].values
Age_data = pd.read_csv('Age_Truth.csv')['Individual Age'].values

College_ranking = np.argsort( np.argsort(-College_data) ) + 1
Age_ranking = np.argsort( np.argsort(-Age_data) ) + 1



# Training
WorldCollege_model_1 = hodge_rank('WorldCollege_data.csv')
WorldCollege_model_2 = hodge_rank('WorldCollege_data.csv')
WorldCollege_model_3 = hodge_rank('WorldCollege_data.csv')
WorldCollege_model_4 = hodge_rank('WorldCollege_data.csv')

HumanAge_model_1 = hodge_rank('HumanAge_data.csv')
HumanAge_model_2 = hodge_rank('HumanAge_data.csv')
HumanAge_model_3 = hodge_rank('HumanAge_data.csv')
HumanAge_model_4 = hodge_rank('HumanAge_data.csv')

WorldCollege_model_1.train(model = 'Uniform')
WorldCollege_model_2.train(model = 'Bradley-Terry')
WorldCollege_model_3.train(model = 'Thurstone-Mosteller')
WorldCollege_model_4.train(model = 'Angular transform')

HumanAge_model_1.train(model = 'Uniform', curl_proj = True)
HumanAge_model_2.train(model = 'Bradley-Terry', curl_proj = True)
HumanAge_model_3.train(model = 'Thurstone-Mosteller', curl_proj = True)
HumanAge_model_4.train(model = 'Angular transform', curl_proj = True)



# Table1
table1 = pd.DataFrame( np.zeros( (2, 4) ) )
table1.index = ['kendalltau', 'Inc.Total']
table1.columns = ['Uniform', 'Bradley-Terry', 'Thurstone-Mosteller', 'Angular transform']
table1.iloc[0] = [stats.kendalltau(WorldCollege_model_1.ranking, College_ranking, method='exact')[0],
                  stats.kendalltau(WorldCollege_model_2.ranking, College_ranking, method='exact')[0],
                  stats.kendalltau(WorldCollege_model_3.ranking, College_ranking, method='exact')[0],
                  stats.kendalltau(WorldCollege_model_4.ranking, College_ranking, method='exact')[0]]
table1.iloc[1] = [WorldCollege_model_1.Cp, WorldCollege_model_2.Cp, WorldCollege_model_3.Cp, WorldCollege_model_4.Cp]

print(table1)



# Table2
eps = 1e-15
table2 = pd.DataFrame( np.zeros( (3, 4) ) )
table2.index = ['kendalltau', 'Inc.Curl', 'Inc.Harm']
table2.columns = ['Uniform', 'Bradley-Terry', 'Thurstone-Mosteller', 'Angular transform']
table2.iloc[0] = [stats.kendalltau(HumanAge_model_1.ranking, Age_ranking, method='exact')[0],
                  stats.kendalltau(HumanAge_model_2.ranking, Age_ranking, method='exact')[0],
                  stats.kendalltau(HumanAge_model_3.ranking, Age_ranking, method='exact')[0],
                  stats.kendalltau(HumanAge_model_4.ranking, Age_ranking, method='exact')[0]]
table2.iloc[1] = [HumanAge_model_1.Inc_curl, HumanAge_model_2.Inc_curl, HumanAge_model_3.Inc_curl, HumanAge_model_4.Inc_curl]
table2.iloc[2] = [HumanAge_model_1.Inc_harm * (HumanAge_model_1.Inc_harm >= eps), 
                  HumanAge_model_2.Inc_harm * (HumanAge_model_2.Inc_harm >= eps),
                  HumanAge_model_3.Inc_harm * (HumanAge_model_3.Inc_harm >= eps),
                  HumanAge_model_4.Inc_harm * (HumanAge_model_4.Inc_harm >= eps)]

print(table2)



# Figure1
plt.subplot(1, 2, 1)
plt.scatter(College_ranking, WorldCollege_model_1.ranking)
plt.xlabel('Ranking of College Score')
plt.ylabel('Ranking of HodgeRank')

plt.subplot(1, 2, 2)
plt.scatter(Age_ranking, HumanAge_model_1.ranking)
plt.xlabel('Ranking of Age-Groundtruth')
plt.savefig('Figure1.png')


# Table3
Max_College = heapq.nlargest(5, enumerate(WorldCollege_model_1.cr), key = lambda x:x[1])
tri_college = WorldCollege_model_1.sparse_data[4]
table3 = pd.DataFrame( np.zeros( (5, 2) ) )
table3.columns = ['Triangle', 'Cr']
A, B = [], []
for k,ind in enumerate(Max_College):
    A.append( ( tri_college[ ind[0] ][0] + 1, tri_college[ ind[0] ][1] + 1, tri_college[ ind[0] ][2] + 1 ) )
    B.append( ind[1] )
table3['Triangle'] = A
table3['Cr'] = B

print(table3)



# Table4
Max_Age = heapq.nlargest(5, enumerate(HumanAge_model_1.cr), key = lambda x:x[1])
tri_age = HumanAge_model_1.sparse_data[4]
table4 = pd.DataFrame( np.zeros( (5, 2) ) )
table4.columns = ['Triangle', 'Cr']
A, B = [], []
for k,ind in enumerate(Max_Age):
    A.append( ( tri_age[ ind[0] ][0] + 1, tri_age[ ind[0] ][1] + 1, tri_age[ ind[0] ][2] + 1 ) )
    B.append( ind[1] )
table4['Triangle'] = A
table4['Cr'] = B

print(table4)



# Figure2
WorldCollege_model_1.train_individual()
Indi_College = heapq.nlargest(8, enumerate(WorldCollege_model_1.Cp_Individual), key = lambda x:x[1])
Ind_Indi_College = np.zeros(8)
Inc_Indi_College = np.zeros(8)
for k,ind in enumerate(Indi_College):
    Ind_Indi_College[k] = ind[0] + 1 
    Inc_Indi_College[k] = ind[1]
Ind_Indi_College = Ind_Indi_College.astype(int).astype(str)

HumanAge_model_1.train_individual()
Indi_Age = heapq.nlargest(8, enumerate(HumanAge_model_1.Cp_Individual), key = lambda x:x[1])
Ind_Indi_Age = np.zeros(8)
Inc_Indi_Age = np.zeros(8)
for k,ind in enumerate(Indi_Age):
    Ind_Indi_Age[k] = ind[0] + 1 
    Inc_Indi_Age[k] = ind[1]
Ind_Indi_Age = Ind_Indi_Age.astype(int).astype(str)

plt.subplot(1, 2, 1)
plt.bar(Ind_Indi_College, Inc_Indi_College)
plt.xlabel('Assessor ID')
plt.ylabel('Inc.Total')

plt.subplot(1, 2, 2)
plt.bar(Ind_Indi_Age, Inc_Indi_Age)
plt.xlabel('Assessor ID')
plt.savefig('Figure2.png')