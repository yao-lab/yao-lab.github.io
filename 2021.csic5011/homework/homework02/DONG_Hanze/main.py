import scipy.io
import numpy as np

data = scipy.io.loadmat('snp452-data.mat')
X = np.array(data['X'])
Y = np.log(X)
print(Y.shape)
deltaY = Y[1:] - Y[:-1]
print(deltaY.shape)
t = deltaY.shape[0]
dim = deltaY.shape[1]
sigma_hat = np.cov(deltaY.T)
print(sigma_hat.shape)
eig_val,_ = np.linalg.eig(sigma_hat)
eig_val= np.sort(eig_val.astype('float64'))[::-1]

R = 100
N_full = np.zeros(eig_val.shape)
for i in range(R):
    deltaY_per = np.zeros(deltaY.shape)
    idx = np.array([np.arange(t)]+[np.random.permutation(t) for i in range(dim-1)])
    #print(idx.shape)
    for j in range(dim):
        deltaY_per[:,j] = deltaY[:,j][idx[j]]
    sigma_hat_per = np.cov(deltaY_per.T)
    eig_val_per,_ = np.linalg.eig(sigma_hat_per)
    eig_val_per= np.sort(eig_val_per.astype('float64'))[::-1]
    N = ((eig_val-eig_val_per)<0).astype('float64')
    N_full+=N
N_full+=1
N_full/=R+1
print(N_full)
# it seems that k = 13


