# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:17:20 2023

@author: LI YAKUN /HW1 FOR CSIC 5011/ MATH 5473
"""

import numpy as np
import numpy.linalg as alg
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.cm as cm
N = 643
p = 256
data = pd.read_csv(r'D:\documents\ust\course\5473\HW1_LI YAKUN\HW1_T1_set.csv')
X = np.array(data)
X = X.T

mu = np.mean(X, axis=0)
print('The sample mean is', mu, '\n')

scalar = StandardScaler()
scalar.fit(X)
X = scalar.transform(X)
print(X, '\n')

u, s, v = alg.svd(X)

print('The shape of u is ',u.shape)
print('The shape of v is ',v.shape, '\n')

[print('The top {}/20 eigen value of SVD is {}'.format(i+1, s[i])) for i in range(20)]

cov_matrix = np.cov(X)
print(cov_matrix, '\n')

eigen_values, eigen_vectors = alg.eig(cov_matrix)
print(eigen_values, '\n')

eigen_pairs = [ (np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]

eigen_pairs.sort(key = lambda eigen_pairs: eigen_pairs[0])
eigen_pairs.reverse()

t = [i for i in range(len(eigen_values))]
plt.plot(t, eigen_values)
plt.show()

total = sum(eigen_values)

var_exped_np = [(i/total) for i in sorted(eigen_values, reverse=True)]
cum_var_exped_np = np.cumsum(var_exped_np)

fig = plt.figure()
ax = fig.subplots(1, 1)
ax.scatter(range(1, len(cum_var_exped_np)+1), cum_var_exped_np)

plt.xlabel = ('Number of Principal Components')
plt.ylabel = ('Cumulatice Explained Variance in Percent')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.show()

plt.figure()
cmap = cm.gray_r
for i in range(1, 21):
    pc = u[:, i]
    pc_matrix = np.reshape(pc, (16, 16))

    plt.subplot(4, 5, i)
    plt.imshow(pc_matrix, cmap=cmap)
    plt.xticks([])
    plt.yticks([])

plt.show()

v1 = eigen_vectors[:, 0]
proj1 = np.dot(X.T, v1)
proj1_tuple = [(i, proj1[i]) for i in range(N)]
proj1_tuple.sort(key= lambda proj1_tuple: proj1_tuple[1])
print("Let's have a look at the rank of images projected on the first principal component\n")
print("The smallest images are:")
[print("The {}/643 samllest image is {}-th with value {} on projection".format(i+1, proj1_tuple[i][0], proj1_tuple[i][1])) for i in range(10)]
print('\n')
print("The biggest images are:")
print("The length of proj1_tuple is {}".format(len(proj1_tuple)))
[print("The {}/643 samllest image is {}-th with value {} on projection".format(i+1, proj1_tuple[N-i-1][0], proj1_tuple[N-i-1][1])) for i in range(10)]

pc1 = eigen_pairs[0][1][:, np.newaxis]
pc2 = eigen_pairs[1][1][:, np.newaxis]

W = np.hstack((pc1, pc2))
print(W.shape)
print(X.shape)
X_pca = np.dot(X.T, W)

plt.figure(figsize=(10, 10))
x = np.array(X_pca[:, 0])
y = np.array(X_pca[:, 1])
plt.scatter(x, y)
plt.xlabel = ('pca1')
plt.ylabel = ('pca2')
plt.title = ('Samples projected onto 2-D by PCA')
plt.show()
