# author: LIU Yunqin
# student ID: 21073799

import numpy as np

n = 3000
p = 1000
gama = p/n
lambda_0_list = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
lambda_list = []
eigenvector_list = []


#calculate the largest eigenvalue when lambda0 in different levels
for lambda_0 in lambda_0_list:
    X = np.zeros((p,n))
    X[0] = np.random.randn(n) * np.sqrt(lambda_0 + 1)
    X[1:,] = np.random.randn(p - 1, n)
    sample_Sigma = X.dot(X.T)/n
    lmd, V = np.linalg.eigh(sample_Sigma)
    lambda_list.append(lmd[-1])
    eigenvector_list.append(V[:,-1])

squared_corr_list = [eigenvector[0]**2 for eigenvector in eigenvector_list]

# calculate the theoretical lambda:
lambda_theoretical_list = [(1 + np.sqrt(gama)) ** 2 if lmd <= np.sqrt(gama) else (1 + lmd) * (1 + gama / lmd) for
                          lmd in lambda_0_list]
squared_corr_theoretical_list = [0 if lmd <= np.sqrt(gama) else (1-gama/lmd**2)/(1+gama/lmd) for lmd in lambda_0_list]

#calculate the differnece
lambda_diff = np.array(lambda_list)-np.array(lambda_theoretical_list)
squared_corr_diff = np.array(squared_corr_list)-np.array(squared_corr_theoretical_list)


print("the largest eigenvalue when lambda0 in different levels",lambda_list)
print("the theoretical lambda:",lambda_theoretical_list)
print("the difference:",lambda_diff)

print("the squared correlation when lambda0 in different levels",squared_corr_list)
print("the theoretical squared correlation:",squared_corr_theoretical_list)
print("the difference:",squared_corr_diff)
