import numpy as np
import pandas as pd
import numpy.linalg as alg
import scipy as sp
import matplotlib.pyplot as plt

data = pd.read_csv(r'D:\博一下学期\MATH5473-main\MATH5473-main\HW1\CityDistance.csv')
print(data)

Data = np.array(data)
n = len(Data)
cities = np.array(data.columns)

def MDS(D, dim=[]):
    H = -np.ones((n, n))/n
    H = -H.dot(D ** 2).dot(H)/2
    evals, evecs = alg.eigh(H)

    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    w, = np.where(evals > 0)
    if dim!=[]:
        arr = evals
        w = arr.argsort()[-dim:][::-1]
    if np.any(evals[w]<0):
        print('Error: Not enough positive eigenvalues for the selected dim.')
        return []
    L = np.diag(np.sqrt(evals[w]))
    V = evecs[:, w]
    Y = V.dot(L)
    return Y, evals, evecs

X2, eigen_values, eigen_vectors = MDS(Data, dim=2)

total = np.sum(eigen_values)
normed_eigen_values = eigen_values/total
print("Eigenvalues are:\n",eigen_values)
print("Normed Eigenvalues are:\n", normed_eigen_values)

plt.figure()
plt.plot(range(n), normed_eigen_values)
plt.show()

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
for i in range(n):
    plt.scatter(X2[i, 0], X2[i, 1], alpha=.8, label=cities[i])
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('MDS of City dataset')
plt.show()
