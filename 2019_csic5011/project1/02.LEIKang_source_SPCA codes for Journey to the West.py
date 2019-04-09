# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:53:01 2019
SPCA codes

@author: KL
"""

import numpy as np
import pandas as pd
import pyreadr
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler


#convert RData to pd DataFrame
readin = pyreadr.read_r('C:/Users/TW/Downloads/west.RData')
westdf = readin["west"]
chapter = westdf[['chapter']]


#propossesing the data
westdf = westdf.drop(['chapter'],axis=1)    #delete 'chapter' column   408*302
x = westdf.loc[:,:].values
x = StandardScaler(with_std=False).fit_transform(x)    #centerlize the data


#SparsePCA transform
transformer = SparsePCA(n_components=3,\
                        alpha=0.1,\
                        normalize_components=True,\
                        random_state=0)
x_transformed = transformer.fit_transform(x)



# for data analysis
x_transformed.shape
transformer.alpha
egienvetors = transformer.components_
transformer.error_
transformer.get_params(deep=True)
np.mean(transformer.components_==0)
westspca = pd.DataFrame(data = egienvetors, columns=westdf.columns)
Spca1 = westspca.sort_values(by=[0],axis=1)
Spca2 = westspca.sort_values(by=[1],axis=1)
Spca3 = westspca.sort_values(by=[2],axis=1)
Spca4 = westspca.sort_values(by=[3],axis=1)
Spca5 = westspca.sort_values(by=[4],axis=1)

westtrans = pd.DataFrame(data = x_transformed)
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









