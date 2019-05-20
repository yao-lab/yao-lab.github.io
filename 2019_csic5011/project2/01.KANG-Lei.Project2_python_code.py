# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:10:12 2019


# need to install pydiffmap package
You can refer to 
https://pydiffmap.readthedocs.io/en/master/


@author: KL
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from pydiffmap import diffusion_map as dm    # need to install pydiffmap package
from sklearn import manifold
from scipy.io import loadmat    # to load matlab data


#Data proposessing
X = loadmat("C:/Users/TW/Desktop/face.mat")   #Load data
Y = X['Y']
Y = np.reshape(Y,(10304, 33))
Y = Y.T
Y = np.float64(Y)    # take out the face data and reshape it for Diffusimap calculation 


n_points = 33
n_neighbors = 5
n_components = 2
fig = plt.figure(figsize=(15, 8))
plt.suptitle("Manifold Learning with %i points, %i neighbors"
             % (33, n_neighbors), fontsize=14)


###############################################################################
#Diffusion map

t0 = time()
neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}
mydmap = dm.DiffusionMap.from_sklearn(n_evecs=2, k=200, epsilon='bgh', alpha=1.0, neighbor_params=neighbor_params)
Y1 = mydmap.fit_transform(Y)
t1 = time()

print("Diffusion map: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(141)
plt.scatter(Y1[:, 0], Y1[:, 1], cmap=plt.cm.Spectral)
plt.title("Diffusion map (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

###############################################################################
#MDS
t0 = time()
mds = manifold.MDS(n_components, max_iter=2000, n_init=1)
Y2 = mds.fit_transform(Y)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(142)
plt.scatter(Y2[:, 0], Y2[:, 1], cmap=plt.cm.Spectral)
plt.title("MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')



###############################################################################
#Isomap
t0 = time()
Y3 = manifold.Isomap(n_neighbors, n_components).fit_transform(Y)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(143)
plt.scatter(Y3[:, 0], Y3[:, 1], cmap=plt.cm.Spectral)
plt.title("Isomap (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')



###############################################################################
# LLE
t0 = time()
Y4 = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method='standard').fit_transform(Y)
t1 = time()
print("%s: %.2g sec" % ('standard', t1 - t0))
ax = fig.add_subplot(144)
plt.scatter(Y4[:, 0], Y4[:, 1], cmap=plt.cm.Spectral)
plt.title("%s (%.2g sec)" % ('LLE', t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
plt.figure()



###############################################################################
#Diffusion map
# Plot all the images according to the order from dmapsort
#sort the diffusion map by the first eigenvecor
Y1sort = np.argsort(Y1[:,0])
j=1
for i in Y1sort:
    t = Y[i].reshape(112, 92)
    plt.subplot(1,33,j,frameon=False)
    plt.imshow(t)
    j=j+1
    plt.axis('off')
plt.axis('off')
plt.figure()

#MDS
plt.figure()
j=1
#sort the MDS by the first eigenvecor and display all the images
Y2sort = np.argsort(Y2[:,0])
plt.figure()
j=1
for i in Y2sort:
    t = Y[i].reshape(112, 92)
    plt.subplot(1,33,j,frameon=False)
    plt.imshow(t)
    j=j+1
    plt.axis('off')
plt.axis('off')
plt.figure()
#sort the Isomap by the first eigenvecor and display the images
Y3sort = np.argsort(Y3[:,0])
plt.figure()
j=1
for i in Y3sort:
    t = Y[i].reshape(112, 92)
    plt.subplot(1,33,j,frameon=False)
    plt.imshow(t)
    j=j+1
    plt.axis('off')
plt.axis('off')
plt.figure()
#sort the LLE by the first eigenvecor and display the images
Y4sort = np.argsort(Y3[:,0])
plt.figure()
j=1
for i in Y4sort:
    t = Y[i].reshape(112, 92)
    plt.subplot(1,33,j,frameon=False)
    plt.imshow(t)
    j=j+1
    plt.axis('off')
plt.axis('off')
plt.figure()