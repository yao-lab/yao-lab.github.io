import numpy as np
from numpy.linalg import eig

# Set parameters
p = 100    # dimensionality of the data
n = 200    # sample size
sigma = 1    # noise level
u = np.ones(p)    # true signal component
u /= np.linalg.norm(u)
lambd0 = 3    # signal-to-noise ratio

# Generate random data
X = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma**2*np.eye(p) + lambd0*np.outer(u, u), size=n)

# Compute sample covariance matrix and its eigenvalues/eigenvectors
S = np.cov(X.T)
lambdas, V = eig(S)
idx = np.argsort(lambdas)[::-1]
lambdas = lambdas[idx]
V = V[:, idx]
lambda_max = lambdas[0]
v_max = V[:, 0]

# Compute true eigenvalues/eigenvectors
true_lambdas = np.array([sigma**2, lambd0 + sigma**2*(p-1)/p])
true_V = np.column_stack((np.ones(p)/np.sqrt(p), np.sqrt(p/(p-1))*u))

# Compare results
print('Largest eigenvalue (estimated):', lambda_max)
print('Largest eigenvalue (true):', true_lambdas[1])
print('Squared correlation (estimated):', np.abs(np.dot(u, v_max))**2)
print('Squared correlation (true):', np.abs(np.dot(true_V[:, 1], u))**2)
