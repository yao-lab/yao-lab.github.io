import time
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

SNP_Raw = np.load('SNP_Raw.npy').transpose()
Nation_Raw = np.load('Nation_Raw.npy', allow_pickle=True)
Nations = list(set(Nation_Raw))
lw = 1.5
colors = ['blue', 'orange', 'green', 'red', 'brown', 'pink', 'olive']

kernel_pca = KernelPCA(n_components=2, kernel='rbf').fit(SNP_Raw)
n_features = kernel_pca.n_features_in_
GAMMA = [1e-3*1/n_features, 1e-2*1/n_features, 1e-1*1/n_features, 1/n_features, 1e1*1/n_features,
         1e2*1/n_features, 1e3*1/n_features, 1e4*1/n_features, 1e5*1/n_features, 1e6*1/n_features]

for j in range(0, len(GAMMA)):
    starttime = time.perf_counter()
    kernel_pca = KernelPCA(n_components=2, kernel='rbf', gamma=GAMMA[j]).fit(SNP_Raw)
    endtime = time.perf_counter()
    T = kernel_pca.transform(SNP_Raw)
    dataframe = pd.DataFrame({'X': T[:, 0], 'Y': T[:, 1], 'Gamma': GAMMA[j], 'CPU Time': (endtime - starttime)})
    dataframe.to_csv(f'..\\Preliminary_Result\\KernelPCA_Gamma_1e{j-3}_Feature_Inverse.csv', index=False, sep=',')
    plt.figure()
    legend_used = []
    for i in range(0, T.shape[0]):
        i_nation = Nation_Raw[i]
        i_color = colors[Nations.index(i_nation)]
        if i_nation in legend_used:
            plt.scatter(T[i, 0], T[i, 1], color=i_color, alpha=0.8, lw=lw)
        else:
            plt.scatter(T[i, 0], T[i, 1], color=i_color, alpha=0.8, lw=lw, label=i_nation)
            legend_used.append(i_nation)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.savefig(f'..\\Preliminary_Result\\KernelPCA_Gamma_1e{j-3}_Feature_Inverse.png', dpi=600)
