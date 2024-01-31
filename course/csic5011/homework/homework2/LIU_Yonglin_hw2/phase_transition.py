import math
import numpy as np
import pandas as pd
import numpy.linalg as alg
import matplotlib.pyplot as plt

p = 100
n = 500
gamma = p/n
sigma = 1
u = np.ones(p)/p
Ip = np.eye(p)
zero_mean = np.zeros(p)
iteration = 2000
Var_signal = [i/iteration for i in range(1, iteration+1)]
S = np.zeros((p, p))
L = []
V = np.zeros(p)

for i, v in enumerate(Var_signal):
    # For each sample
    for j in range(n):
        alpha = np.random.normal(loc=0, scale=math.sqrt(v), size=1)
        t = alpha * u
        epsilon = np.random.multivariate_normal(zero_mean, Ip, 1)
        x = t + epsilon
        # Compute the Covariance Matrix
        S += 1 / n * np.dot(x, x.T)

    # Do EVD
    eigen_values, eigen_vectors = alg.eig(S)
    eigen_pairs = [(eigen_values[i], eigen_vectors[:, i]) for i in range(len(eigen_values))]
    # Sort the eigen_pairs by eigen_value in decreasing order
    eigen_pairs.sort(key=lambda eigen_pairs: eigen_pairs[0], reverse=True)
    # Find the max eigenvalue and corresponding eigenvector
    lambda_max = eigen_pairs[0][0]
    vector_max = eigen_pairs[0][1]

    if i % 10 == 0:
        print("The {}-th trial with data signal variance {}".format(i + 1, v))
        print("The max eigen value is {}".format(lambda_max))
        print("The correspondign eigen vector is\n {}".format(vector_max))
        print("\n")

    # Record the max eigenvalue and corresponding eigenvector
    L.append(lambda_max)
    if V.all() == 0:
        V = vector_max
    else:
        V = np.hstack((V, vector_max))

V = V.astype(np.float64)

