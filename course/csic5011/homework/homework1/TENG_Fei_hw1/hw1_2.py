# coding:utf-8
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

'''
author: heucoder
email: 812860165@qq.com
date: 2019.6.13
'''

def cal_pairwise_dist(x):
    '''计算pairwise 距离, x是matrix
    (a-b)^2 = a^2 + b^2 - 2*a*b
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #返回任意两个点之间距离的平方
    return dist


def my_mds(data, n_dims):
    '''
    :param data: (n_samples, n_features)
    :param n_dims: target n_dims
    :return: (n_samples, n_dims)
    '''

    n, d = data.shape
    # print(n,d)
    # dist = cal_pairwise_dist(data)
    
    H = np.eye(n) - 1 / n
    D2 = data ** 2
    
    T = -0.5 * np.dot(np.dot(H,D2),H)
    eig_val, eig_vector = np.linalg.eig(T)
    print(-eig_val)
    index_ = np.argsort(-eig_val)[:n_dims]
    picked_eig_val = eig_val[index_].real
    picked_eig_vector = eig_vector[:, index_]
    # print(picked_eig_vector.shape, picked_eig_val.shape)
    return picked_eig_vector*picked_eig_val**(0.5)

if __name__ == '__main__':
    # iris = load_iris()
    # data = iris.data
    # Y = iris.target
    
    # data_1 = my_mds(data, 2)
    data = pd.read_csv('data.csv',header=None,names=['Shanghai','Tianjin','Hong Kong','Los Angeles','Chengdu','London','Thành phố Thanh Hóa'])
    # print(data)
    data_1 = my_mds(data, 2)
    print(data_1)
    # data_2 = MDS(n_components=2).fit_transform(data)
    data_2 = my_mds(data, 3)
    print(data_2)
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title("top_2")
    plt.scatter(data_1[:, 0], data_1[:, 1], c='r')
    for i,name in enumerate(['Shanghai','Tianjin','Hong Kong','Los Angeles','Chengdu','London','Thành phố Thanh Hóa']):
        plt.annotate(name,(data_1[i][0],data_1[i][1]))
    plt.subplot(122)
    plt.title("top_3")
    plt.scatter(data_2[:, 0], data_2[:, 1], c='r')
    
    for i,name in enumerate(['Shanghai','Tianjin','Hong Kong','Los Angeles','Chengdu','London','Thành phố Thanh Hóa']):
        plt.annotate(name,(data_2[i][0],data_2[i][1]))
    plt.savefig("MDS_1.png")
    plt.show()