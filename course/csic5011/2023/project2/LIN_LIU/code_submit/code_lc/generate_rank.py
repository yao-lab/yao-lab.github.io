import numpy as np
from hodge_rank_algo import hodge_rank
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from utils import plot_path, plot_new
data_type = "college"
# ana_mode = "decompose"

if data_type == "college":
    data = np.load('college/college_data.npy',allow_pickle=True)
    gt = np.load('college/college_gt.npy',allow_pickle=True)

    gt_rank = np.argsort(-gt[:,1].astype(np.int64))
    N=9408
    M=409
    n_nodes=261

elif data_type == "age":
    data = np.load('age/age_data.npy',allow_pickle=True)
    gt = np.load('age/age_gt.npy',allow_pickle=True)
    gt_rank = np.argsort(gt[:,1])
    N=12778
    M=94
    n_nodes=30




# print(gt_rank)
model = hodge_rank(data, N, M,n_nodes)
model.get_graph()
# theta,gamma, gamma_path   = model.get_outlier_lbi_with_knockoffs()
# ssss
# theta,gamma, gamma_path   = model.get_outlier_lbi_with_knockoffs(kappa=5, alpha= 1e-3, max_iter=10000, lam=1)
# theta,gamma, gamma_path   = model.get_outlier_lbi(kappa=5, alpha= 5e-3, max_iter=10000, lam=1)
# theta,gamma, gamma_path   = model.get_outlier_lbi(kappa=5, alpha= 5e-3,max_iter=1000, lam=1)
theta,gamma, gamma_path   = model.get_outlier_lbi_with_knockoffs(kappa=1, alpha= 1e-3,max_iter=10000, lam=1, q=0.2)
non_zero_gamma = [i for i in range(len(gamma)) if gamma[i] != 0]
nzero_gamma = [i for i in range(len(gamma)) if gamma[i] == 0]

print(non_zero_gamma)
print(len(non_zero_gamma))
# plot_path(gamma_path, n=10)
plot_new(non_zero_gamma,nzero_gamma,'college_prefer.csv')
# # print(gamma_path.shape)
# pred_rank = np.argsort(theta)
# print(pred_rank)
# print(stats.kendalltau( pred_rank,gt_rank, variant='b'))
# # theta = model.get_global_rank()
# theta, gamma, gamma_path = model.get_outlier_lbi()
# pred_rank = np.argsort(theta)
# print(stats.kendalltau( pred_rank,gt_rank))
#

# non_zero_gamma = [i for i in range(len(gamma)) if gamma[i] != 0]
# print(non_zero_gamma)




