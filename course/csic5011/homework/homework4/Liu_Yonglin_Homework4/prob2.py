import numpy as np
import math
import random
import pandas as pd
from cvxopt import matrix, solvers
from cvxopt.modeling import variable, op, sum, dot
import matplotlib.pyplot as plt

N = 20
d = 20
K = 20
S = np.zeros((N, K), dtype=float)


def ReLU(a):
    if a == 0:
        return -1
    else:
        return 1


for n in range(1, N + 1):
    A = np.random.normal(loc=0, scale=1, size=(n, d))
    for k in range(1, n + 1):
        for i in range(1, 50 + 1):
            # Make a sparse x0
            x0 = np.zeros(d)
            t = random.sample(range(d), k)
            rand_bino = np.random.binomial(1, 0.5, k)
            result = map(ReLU, rand_bino)
            result_list = list(result)
            x0[t] = result_list
            # Draw a standard Gaussian Random Matrix
            A = np.random.normal(loc=0, scale=1, size=(n, d))

            b = np.dot(A, x0)
            # = [-1 if x0[i]<0 else 1 for i in range(len(x0))]
            A = A.T
            A = matrix(A)
            b = matrix(b)
            # c = matrix(c)
            # Solve the linear programming problem
            x = variable(d)
            op(sum(abs(x)), [dot(A, x) == b]).solve()
            x = np.asarray(x.value)
            x = np.squeeze(x)
            dist = np.sqrt(np.sum(np.square(x - x0)))
            if dist <= 1e-3 and n < 20:
                S[n, k] = S[n, k] + 1

S = S/50
plt.imshow(S, origin='upper', extent=[0, K, 0, N])
plt.xlabel('k')
plt.ylabel('n')
plt.show()
