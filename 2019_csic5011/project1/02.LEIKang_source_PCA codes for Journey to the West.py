# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 22:48:53 2019
PCA codes
@author: KL
"""
import pandas as pd
import pyreadr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#convert RData to pd DataFrame
readin = pyreadr.read_r('C:/Users/TW/Downloads/west.RData')
#print(readin.keys()) # let's check what objects we got
westdf = readin["west"]
#print(westdf)
chapter = westdf[['chapter']]

#propossesing the data

westdf = westdf.drop(['chapter'],axis=1)    #delete 'chapter' column   408*302
x = westdf.loc[:,:].values
x = StandardScaler(with_std=False).fit_transform(x)    #centerlize the data

# do pca transform
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)


# for data analysis and display
variance = pca.explained_variance_
plt.plot(variance)
ratio = pca.explained_variance_ratio_
plt.plot(ratio,'bs')
sum(pca.explained_variance_ratio_)
Pivots=pca.components_
westpca = pd.DataFrame(data = Pivots, columns=westdf.columns)

# these are pivots,aligned by loads on each feature
pca1 = westpca.sort_values(by=[0],axis=1)
pca2 = westpca.sort_values(by=[1],axis=1)
pca3 = westpca.sort_values(by=[2],axis=1)
pca4 = westpca.sort_values(by=[3],axis=1)
pca5 = westpca.sort_values(by=[4],axis=1)


westtrans = pd.DataFrame(data = principalComponents)
westpcascore = pd.concat([chapter, westtrans], axis = 1)
pcascore1 = westpcascore.sort_values(by=[0],axis=0)
pcascore2 = westpcascore.sort_values(by=[1],axis=0)
pcascore3 = westpcascore.sort_values(by=[2],axis=0)
pcascore4 = westpcascore.sort_values(by=[3],axis=0)
pcascore5 = westpcascore.sort_values(by=[4],axis=0)

#transpose for better display, these are scores on each pivots
pcascore1 = pcascore1.T
pcascore2 = pcascore2.T
pcascore3 = pcascore3.T
pcascore4 = pcascore4.T
pcascore5 = pcascore5.T





