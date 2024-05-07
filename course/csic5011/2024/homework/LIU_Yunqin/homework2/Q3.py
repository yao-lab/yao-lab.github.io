# author: LIU Yunqin
# student ID: 21073799

# confirm wigner semi-circle law
import numpy as np
import matplotlib.pyplot as plt

n = 400
W = np.zeros((n,n))
W[np.triu_indices(n)] = np.random.randn(int(n*(n+1)/2)) /(2 * np.sqrt(n))
W += W.T - np.diag(W.diagonal())

lmd, V = np.linalg.eigh(W)
n, bins, patches = plt.hist(lmd, density=True, bins=40)

y = 2 / np.pi * np.sqrt(1 - bins ** 2)
plt.plot(bins, y)
plt.xlabel("eigenvalue")
plt.ylabel("frequency")

plt.show()