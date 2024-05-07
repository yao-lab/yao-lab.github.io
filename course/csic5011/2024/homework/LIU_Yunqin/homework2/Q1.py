# author: LIU Yunqin
# student ID: 21073799

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

data = scio.loadmat('snp452-data.mat')
X = data['X'].T

# take the logarithmic pricres Y
Y = np.log(X)

#calculate logarithmic price jumps
dy = Y[:,1:]-Y[:,:-1]

#construct realized covariance matrix
p,n = dy.shape
CovMatrix = dy.dot(dy.T)/n

#compute the eigenvalue and eigenvector
lmd, vec = np.linalg.eig(CovMatrix)

#horn's parallel analysis
R = 300
pvalue = np.zeros(p,dtype='float32')
dy_random = dy[:,:]
for i in range(R):
    for j in range(p-1):
        np.random.shuffle(dy_random[j+1])
    CovMatrix_random = dy_random.dot(dy_random.T)/n
    lmd_random = np.linalg.eigh(CovMatrix_random)[0][::-1]
    pvalue = pvalue + (lmd < lmd_random)
pvalue = (pvalue + 1)/(R + 1)

plt.plot(pvalue[0:19])
plt.show()

#small p-values are considered to be signal