# coding:utf-8
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

def MDS(data, n_dims):
    # data is a matrix and n_dims is the reducing demention number
    n, _ = data.shape
    H = np.eye(n) - np.ones((n, n))/n
    B = -0.5 * np.dot(np.dot(H, np.power(data, 2)),H)
    eig_value, eig_vector = np.linalg.eig(B)
    idx = np.argsort(-eig_value)[:n_dims]
    eigen_value = eig_value[idx]
    # check eigen values
    print([e/np.sum(eigen_value)for e in eigen_value])
    eigen_vector = eig_vector[:, idx]
    result = np.dot(eigen_vector, np.sqrt(np.diag(eigen_value)))
    return result

if __name__ == '__main__':
    data = [l[1:] for l in pd.read_csv('data.csv').values.tolist()]
    data_mds = MDS(np.array(data), 2)
    plt.title("top_2")
    plt.scatter(data_mds[:, 0], data_mds[:, 1], c='r')
    for i,name in enumerate(['Shanghai','Beijing','Wuhan','Chongqing','Hangzhou','Chengdu','Tianjin']):
        plt.annotate(name,(data_mds[i][0],data_mds[i][1]))
    plt.show()