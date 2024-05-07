import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# (a) input a few cities
data = pd.read_csv('city_distance.csv')

# (b) MDS
k = 2
n = data.shape[0]
H = np.eye(n) - 1/n*np.ones((n,n))
B = -1/2*H*data*np.transpose(H)
u, s, vh = np.linalg.svd(B)
lambda_k = np.diag(s[:k])
u_k = u[:,:k]
X_tidle = np.dot(np.sqrt(lambda_k),np.transpose(u_k))

# (c) Plot the normalized eigenvalues
plt.plot(s/s.sum())
plt.show()
plt.scatter(X_tidle[0], X_tidle[1], s=100, lw=0, label="MDS")
cities = ['Beijing', 'Shanghai', 'HongKong', 'Wuhan', 'Shenyang', 'Chengdu', 'Haikou']
for i, txt in enumerate(cities):
    plt.annotate(txt, (X_tidle[0, i], X_tidle[1, i]))

plt.xticks([])
plt.yticks([])
plt.show()