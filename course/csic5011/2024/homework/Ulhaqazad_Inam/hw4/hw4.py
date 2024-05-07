# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:31:25 2024

@author: iuaa
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
#%%

path = r"D:\OneDrive - HKUST Connect\CSIC5011\HW4\ceph_hgdp_minor_code_XNA.betterAnnotated.csv\ceph_hgdp_minor_code_XNA.betterAnnotated.csv"
path2 = r"D:\OneDrive - HKUST Connect\CSIC5011\HW4\ceph_hgdp_minor_code_XNA.sampleInformation.csv"

df = pd.read_csv(path)

dfr = pd.read_csv(path2)

snp = df.iloc[:, 3:]
#%%  find the random projections and bernoulli random matrix

p,n = snp.shape

k = 1000 # reduce the dimensions

R = np.zeros((k,p))

for i in range(k):
    j = np.random.randint(0,p, size=k)
    R[i,j] = 1/k

H = np.eye(n) - (1/n)*np.ones((n,n))

X = np.array(snp)

XL = np.dot(X.T,R.T)

P = np.dot(H,XL)

K = np.dot(P,P.T) # find the K matrix

eigenvalue, eigenvector = np.linalg.eig(K) # find eigen values and eigen vector

pc1 = np.sqrt(eigenvalue[0]* eigenvector[:,0])
pc2 = np.sqrt(eigenvalue[1]* eigenvector[:,1])


#%% find unique region
regions = dfr['region'].unique()

#%% find random projection for pca and mds 

random_projection = GaussianRandomProjection(n_components=500)  
X_random_projection = random_projection.fit_transform(snp.T)

# Apply PCA 
pca = PCA(n_components=2)  
X_pca = pca.fit_transform(X_random_projection)  

# Apply MDS 
mds = MDS(n_components=2)  
X_mds = mds.fit_transform(X_random_projection)  

# plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.title('PCA')

plt.subplot(1, 2, 2)
plt.scatter(X_mds[:, 0], X_mds[:, 1], alpha=0.5)
plt.title('MDS')

plt.show()

#%% regional data
regionid = []
for region in regions:
    ids = dfr[dfr['region'] == region].index
    regionid.append(ids)

fig = plt.figure(figsize=(6, 6))

for i, region in enumerate(regions):
    plt.scatter(X_pca[regionid[i], 0], X_pca[regionid[i], 1], label=region)

plt.legend()
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

#%% Q-2

import cvxpy as cp

def generate_sparse_vector(d, k):
    x0 = np.zeros(d)
    nonzero_indices = np.random.choice(d, k, replace=False)
    x0[nonzero_indices] = np.random.choice([-1, 1], k)
    return x0

def compute_success_probability(n, d, k, num_trials=50, threshold=1e-3):
    success_count = 0
    for _ in range(num_trials):
        # sparse vector
        x0 = generate_sparse_vector(d, k)
        
        # Gaussian random matrix A
        A = np.random.randn(n, d)
        
        #  b = Ax0
        b = np.dot(A, x0)
        
        # linear programming
        x_hat = cp.Variable(d)
        objective = cp.Minimize(cp.norm(x_hat, 1))
        constraints = [A @ x_hat == b]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        x_hat_value = x_hat.value
        
        # Step 5: Check success criterion
        if np.linalg.norm(x_hat_value - x0, ord=1) <= threshold:
            success_count += 1
    
    
    return success_count / num_trials


d = 20  # Dimensionality 
max_n = d  # Maximum number of measurements
max_k = d  # Maximum sparsity level

# Initialize success probability matrix
success_probabilities = np.zeros((max_n, max_k))

# Iterate over different values of n and k
for n in range(1, max_n + 1):
    for k in range(1, max_k + 1):
        success_probabilities[n - 1, k - 1] = compute_success_probability(n, d, k)

# Visualize success probability as a heatmap
plt.imshow(success_probabilities,  aspect='auto', cmap='Greens', origin='lower', vmin=0, vmax=1)
plt.colorbar(label='Success Probability')
plt.xlabel('Sparsity Level (k)')
plt.ylabel('Number of Measurements (n)')
plt.title('Success Probability vs. Sparsity Level and Number of Measurements')
plt.show()

