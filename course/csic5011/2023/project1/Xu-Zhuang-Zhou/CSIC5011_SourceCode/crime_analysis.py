import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA 
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler

import os

parameters = [1,2,3,4] # criminal risk level
crime_types = ['murder', 'rape', 'robbery', 'assault',	'burglary',	'larceny',	'auto','total_crime']
vis_col = 'total_crime'
pearson_coff = 0.3

if not os.path.exists('./figs_{}/'.format(pearson_coff)):
    os.makedirs('./figs_{}/'.format(pearson_coff))

def plotting2Dgraph(finalDF,col,title):
    parameters = [1,2,3,4]
    figure = plt.figure()
    ax = figure.add_subplot()
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title(title,fontsize = 15)
    for parameter in parameters:
        indicesToKeep = finalDF[col+'_level'] == (parameter-1)
        ax.scatter(finalDF.loc[indicesToKeep, 'principal component 1']
               , finalDF.loc[indicesToKeep, 'principal component 2'])
    ax.legend(parameters)
    ax.grid()
    figure.savefig('./figs_{}/'.format(pearson_coff)+col+'_'+ title + '.png')

crime = pd.read_csv('./crime data/crime_filter_pearson_{}.csv'.format(pearson_coff)) #loading the preprocessed data
print("n_features: ",len(crime.drop(labels=crime_types, axis=1).iloc[:,:-8].columns))
crime_matrix = crime.drop(labels=crime_types, axis=1).iloc[:,:-8].values #obtain all data from the selected parameters
crime_matrix = StandardScaler().fit_transform(crime_matrix)


# visualizing the 2D projection
# PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(crime_matrix)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, crime[vis_col+'_level']], axis = 1)
print('components:')
print(pca.components_)
print('explained_variance_ratio: ', pca.explained_variance_ratio_)
print(finalDf)
plotting2Dgraph(finalDf, vis_col, 'PCA 2 Components Result')

#spca
spca = SparsePCA(n_components=2,random_state=0)
sprincipalComponents = spca.fit_transform(crime_matrix)
sprincipalDf = pd.DataFrame(data = sprincipalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([sprincipalDf, crime[vis_col+'_level']], axis = 1)
print('components:')
print(spca.components_)
print(finalDf)
plotting2Dgraph(finalDf, vis_col, 'SparePCA 2 Components Result')

#mds
mds = MDS(n_components=2)
MprincipalComponents = mds.fit_transform(crime_matrix)
MprincipalDf = pd.DataFrame(data = MprincipalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([MprincipalDf, crime[vis_col+'_level']], axis = 1)
print(finalDf)
plotting2Dgraph(finalDf, vis_col, 'MDS 2 Components Result')

#lle
lle = LocallyLinearEmbedding(n_components=2)
LprincipalComponents = lle.fit_transform(crime_matrix)
LprincipalDf = pd.DataFrame(data = LprincipalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([LprincipalDf, crime[vis_col+'_level']], axis = 1)
print(finalDf)
plotting2Dgraph(finalDf, vis_col,'LLE 2 Components Result')

#isomap
Isp = Isomap(n_components=2)
iprincipalComponents = Isp.fit_transform(crime_matrix)
iprincipalDf = pd.DataFrame(data = iprincipalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([iprincipalDf, crime[vis_col+'_level']], axis = 1)
print(finalDf)
plotting2Dgraph(finalDf, vis_col,'ISOMAP 2 Components Result')

#pca 3d
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(crime_matrix)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2','principal component 3'])
finalDf = pd.concat([principalDf, crime[vis_col+'_level']], axis = 1)
print(finalDf)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_zlabel('Principal Component 3',fontsize = 10)
ax.set_title('PCA 3 Components Result', fontsize = 20)
parameters=[1,2,3,4]
for parameter in parameters:
    indicesToKeep = finalDf[vis_col+'_level'] == (parameter-1)
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2'],
               finalDf.loc[indicesToKeep, 'principal component 3'])
ax.legend(parameters)
ax.grid()
fig.savefig('./figs_{}/'.format(pearson_coff)+vis_col+'_'+'pca3d.png')

#isomp 3d
Isp = Isomap(n_components=3)
iprincipalComponents = Isp.fit_transform(crime_matrix)
iprincipalDf = pd.DataFrame(data = iprincipalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([iprincipalDf, crime[vis_col+'_level']], axis = 1)
print(finalDf)
fig6 = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_zlabel('Principal Component 3',fontsize = 10)
ax.set_title('ISOMP 3 Components Result', fontsize = 20)

parameters=[1,2,3,4]
for parameter in parameters:
    indicesToKeep = finalDf[vis_col+'_level'] == (parameter-1)
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2'],
               finalDf.loc[indicesToKeep, 'principal component 3'])
ax.legend(parameters)
ax.grid()
fig6.savefig('./figs_{}/'.format(pearson_coff)+vis_col+'_'+'isomap3d.png')

#lle 3d
lle = LocallyLinearEmbedding(n_components=3)
LprincipalComponents = lle.fit_transform(crime_matrix)
LprincipalDf = pd.DataFrame(data = LprincipalComponents, columns = ['principal component 1', 'principal component 2','principal component 3'])
finalDf = pd.concat([LprincipalDf, crime[vis_col+'_level']], axis = 1)
print(finalDf)
fig5 = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_zlabel('Principal Component 3',fontsize = 10)
ax.set_title('LLE 3 Components Result', fontsize = 20)

parameters=[1,2,3,4]
for parameter in parameters:
    indicesToKeep = finalDf[vis_col+'_level'] == (parameter-1)
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2'],
               finalDf.loc[indicesToKeep, 'principal component 3'])
ax.legend(parameters)
fig5.savefig('./figs_{}/'.format(pearson_coff)+vis_col+'_'+'LLE3d.png')

#spca 3d
spca = SparsePCA(n_components=3)
principalComponents = spca.fit_transform(crime_matrix)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2','principal component 3'])
finalDf = pd.concat([principalDf, crime[vis_col+'_level']], axis = 1)
print(finalDf)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_zlabel('Principal Component 3',fontsize = 10)
ax.set_title('SparePCA 3 Components Result', fontsize = 20)

parameters=[1,2,3,4]
for parameter in parameters:
    indicesToKeep = finalDf[vis_col+'_level'] == (parameter-1)
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2'],
               finalDf.loc[indicesToKeep, 'principal component 3'])
ax.legend(parameters)
ax.grid()
fig.savefig('./figs_{}/'.format(pearson_coff)+vis_col+'_'+'spca3d.png')

#mds 3d
mpca = MDS(n_components=3)
principalComponents = mpca.fit_transform(crime_matrix)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2','principal component 3'])
finalDf = pd.concat([principalDf, crime[vis_col+'_level']], axis = 1)
print(finalDf)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_zlabel('Principal Component 3',fontsize = 10)
ax.set_title('MDS 3 Components Result', fontsize = 20)
parameters=[1,2,3,4]
for parameter in parameters:
    indicesToKeep = finalDf[vis_col+'_level'] == (parameter-1)
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2'],
               finalDf.loc[indicesToKeep, 'principal component 3'])
ax.legend(parameters)
ax.grid()
fig.savefig('./figs_{}/'.format(pearson_coff)+vis_col+'_'+'mds3d.png')