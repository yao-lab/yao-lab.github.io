
# coding: utf-8

# # 1

# In[9]:


import numpy as np
from sklearn import random_projection
import matplotlib.pyplot as plt


# In[25]:


from pandas import read_csv
data_path = 'ceph_hgdp_minor_code_XNA.betterAnnotated.csv'
dataFrame = read_csv(data_path, header=0)
data = dataFrame.values
print('finishing reading')    
print(data.shape)


# In[28]:


data1=np.transpose(data[:,3:1046])
data1.shape


# In[29]:


transformer = random_projection.GaussianRandomProjection()
data_reduced = transformer.fit_transform(data1)


# In[30]:


data_reduced.shape


# In[31]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(data_reduced)
data_pca = pca.transform(data_reduced)


# In[37]:


plt.scatter(data_pca[:, 0], data_pca[:, 1])


# # 2

# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install cvxpy')


# In[47]:


import numpy as np
import random
import scipy.sparse as sparse
import cvxpy as cp


# In[42]:




# In[ ]:


d=20
iteration=50
p=[]
n_list=[]
k_list=[]
for n in range (1,d+1):
    for k in range (1,d+1):
        n_list.append[n]
        k_list.append[n]
        for i in range(0, iteration):
            A = np.random.normal(0, 1, size=(n, d))
            print('A shape: ', A.shape)
            p_suc = 0
            non_zero=np.random.choice([-1,1], k, p=[0.5,0.5]).tolist()
            x0 = np.array(non_zero + [0] * (d-k) )
            x0 = x0[np.random.permutation(d) - 1 ]
            print('x0 shape: ', x0.shape)
            b=np.dot(A,x0)
            print('b shape: ', b.shape)
            x_opt=prob.value
            print(x_opt)
            x = cp.Variable(n)
            objective = cp.Minimize(cp.sum(np.abs(x)))
            constraints = [np.dot(A,x)=b]
            prob = cp.Problem(objective, constraints)
            result = prob.solve()
            x_opt=x.value

            if(numpy.linalg.norm(x_opt-x0)<=10**(-3)):
                print('success')
                p_suc=p_suc+1

        print(p_suc)
        p.append[p_suc]
plt.pcolor(n_list, k_list, p)
fig.colorbar(c)                

