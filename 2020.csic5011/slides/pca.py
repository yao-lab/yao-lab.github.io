#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Principal Component Analysis (PCA): an example on dataset zip digit 3
=========================================================

The PCA does an unsupervised dimensionality reduction, as the best affine 
k-space approximation of the Euclidean data.

"""
print(__doc__)


# Created by Yuan YAO, HKUST 
# 6 Feb, 2017

import pandas as pd
import io
import requests

import numpy as np

# Load dataset as 16x16 gray scale images of handwritten zip code 3, of total number 657.

url = "https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.digits/train.3"
s = requests.get(url).content
c = pd.read_csv(io.StringIO(s.decode('utf-8')))
data = np.array(c,dtype='float32');
# data = np.array(pd.read_csv('train.3'),dtype='float32');
data.shape

# Reshape the data into image of 16x16 and show the image.
import matplotlib.pyplot as plt
img1 = np.reshape(data[1,:],(16,16));
imgshow = plt.imshow(img1,cmap='gray')

img2 = np.reshape(data[39,:],(16,16));
imgshow = plt.imshow(img2,cmap='gray')

# Now show the mean image.
mu = np.mean(data, axis=0);
img_mu = np.reshape(mu,(16,16));
imgshow = plt.imshow(img_mu,cmap='gray')

##########################################
# PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=50, svd_solver='arpack')
pca.fit(data)

print(pca.explained_variance_ratio_) 

# Plot the 'explained_variance_ratio_'

plt.plot(pca.explained_variance_ratio_, "o", linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_ratio_')

# Principal components

Y = pca.components_;
Y.shape

# Show the image of the 1st principal component

img_pca1 = np.reshape(Y[1,:],(16,16));
imgshow = plt.imshow(img_pca1,cmap='gray')

# Show the image of the 2nd principal component

img_pca2 = np.reshape(Y[2,:],(16,16));
imgshow = plt.imshow(img_pca2,cmap='gray')

# Show the image of the 3rd principal component

img_pca3 = np.reshape(Y[3,:],(16,16));
imgshow = plt.imshow(img_pca3,cmap='gray')