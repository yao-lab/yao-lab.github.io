import numpy as np
import numpy.linalg as alg
import pandas as pd
import matplotlib.pyplot as plt

p = 452
n = 1258
R = 1000

df = pd.read_csv('snp452-data.csv')
df.head(6)

X = np.asarray(df)
X = X.T
print("Now the shape of X is {}".format(X.shape))

Y = np.log(X)

dY = np.diff(Y, axis = 1)
print("The shape of Y after differentiating is {}".format(dY.shape))
p, n = dY.shape

S = np.cov(dY)
S.shape

eigen_values, eigen_vectors = alg.eig(S)
eigen_pairs = [(eigen_values[i], eigen_vectors[:, i]) for i in range(len(eigen_values))]
eigen_pairs.sort(key=lambda eigen_pairs: eigen_pairs[0], reverse=True)

N = np.zeros(p)

for r in range(R):
    Sr = np.zeros(shape=(p, p))
    # Create a permutated Sr
    for i in range(p):
        permu = np.random.permutation(p)
        Sr[i, :] = S[i, permu]

    Sr_eigen_values, Sr_eigen_vectors = alg.eig(Sr)
    Sr_eigen_pairs = [(Sr_eigen_values[j], Sr_eigen_vectors[:, j]) for j in range(len(Sr_eigen_values))]

    Sr_eigen_pairs.sort(key=lambda Sr_eigen_pairs: Sr_eigen_pairs[0],
                       reverse=True)

    for k in range(p):
        if eigen_values[k] < Sr_eigen_values[k]:
            N[k] += 1

P_values = [(N[k]+1)/(R+1) for k in range(p)]

plt.plot(P_values)
plt.show()

for k in range(p):
    if N[k] > 0:
        print("The first eigenvalue that has bigger competitor from Sr's is the {}-th\n its value is {} with p-value {} ".format(k, N[k], P_values[k]))
        break