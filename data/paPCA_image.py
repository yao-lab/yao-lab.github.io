import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

n_perm = 10 # number of parallel dataset
perc = 0.05 # percentage

################################################################
X = np.loadtxt('train.3', delimiter = ',')

n, dim = X.shape
mean = np.mean(X, axis = 0)
X0 = X - mean
#X0 = X
Cov = np.dot(X0.T, X0)/(n-1)


evals, evecs = np.linalg.eigh(Cov)
evals = evals[::-1]
evecs = evecs[:, ::-1]

modes = np.c_[mean.reshape(-1, 1), evecs]

################################################################
Xcp = X0.copy()
evals_perm = np.zeros([n_perm, dim])

for i in range(n_perm):
    for j in range(1, dim):
        np.random.shuffle(Xcp[:, j])
    Cov_perm = np.dot(Xcp.T, Xcp)/(n-1)
    evals_perm[i] = np.linalg.eigvalsh(Cov_perm)[::-1]

evals0 = np.mean(evals_perm, axis = 0)
evals_perm = np.sort(evals_perm, axis = 0)[::-1]
evals_perc = evals_perm[int(np.floor(perc * n_perm))]
pvals = np.mean((evals_perm > evals).astype(float), axis = 0)

# index of the first nonzero p-values
for j in range(dim):
    if pvals[j] > 0:
        pv1 = j
        break

################################################################
plt.figure(figsize = (20, 10))
ax = plt.subplot(111)
ax.loglog(evals, 'r-o', linewidth = 2, label = r'original')
#ax.loglog(evals0, 'g-*', linewidth = 2, label = r'permuted mean')
ax.loglog(evals_perc, 'y-^', linewidth = 2, label = r'permuted top %s%s'%(perc*100, '%'))
ax.plot(np.nan, 'b', linewidth = 2, label = r'p-value') # agent
ax.set_xticks([pv1+1, dim])
ax.set_xticklabels([pv1+1, dim])
ax.tick_params(axis='x', labelsize = 20)
ax.tick_params(axis='y', labelsize = 20)
ax.set_xlabel('dimensions (log scale)', fontsize = 20)
ax.set_ylabel('eigenvalues', fontsize = 20)
ax.legend(loc = 'lower left', fontsize = 20)

ax1 = ax.twinx()
ax1.plot(pvals, 'b', linewidth = 2)
ax1.vlines(pv1, 0, 1, 'b', 'dashed', linewidth = 2, label = r'1-st nonzero p-values')
ax1.hlines(perc, pv1, dim, 'k', 'dotted', linewidth = 2)
ax1.fill_between(np.arange(dim), np.ones(dim), where = (pvals > perc),
                alpha = 0.2, label = r'color fill: for p-value > %s%s'%(perc*100, '%'))
ax1.tick_params(axis='y', labelsize = 20)
ax1.set_yticks([perc, 1])
ax1.set_yticklabels(['%s%s'%(100*perc, '%'), '%s%s'%(100, '%')])
ax1.set_ylabel('p-values', fontsize = 20)
ax1.legend(loc = 'upper right', fontsize = 20)

#plt.title('parallel analysis of PCA', fontsize = 30)
plt.xlim(0, dim)
plt.show()



################################################################
#img = X[0].reshape(16, 16)
plt.figure(figsize = (10, 10))
#plt.title('mean and principal components', fontsize = 20)
for j in range(25):
#    img = Xcp[j].reshape(16, 16)
    img = modes[:, j].reshape(16, 16)
    ax = plt.subplot(5, 5, j+1)
    ax.imshow(img, cmap = 'gray')
    ax.set_title('%d'%(j+1), fontsize = 20)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()












