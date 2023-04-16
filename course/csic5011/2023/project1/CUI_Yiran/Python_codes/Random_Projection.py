import time
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import random_projection, manifold
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.decomposition import PCA

# ===================================================================================#
LOOP = 2  # 1 for RP-PCA; 2 for RP-MDS
# ===================================================================================#

SNP_Raw = np.load('SNP_Raw.npy').transpose()
Nation_Raw = np.load('Nation_Raw.npy', allow_pickle=True)
Nations = list(set(Nation_Raw))
lw = 1.5
colors = ['blue', 'orange', 'green', 'red', 'brown', 'pink', 'olive']
D_ref = johnson_lindenstrauss_min_dim(n_samples=SNP_Raw.shape[0], eps=0.1)

Dimension = [100, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000]

if LOOP == 1:
    for j in range(0, len(Dimension)):
        projector = random_projection.GaussianRandomProjection(n_components=int(Dimension[j]))
        SNP_RP = projector.fit_transform(SNP_Raw)
        starttime = time.perf_counter()
        T = PCA(n_components=2).fit_transform(SNP_RP)
        endtime = time.perf_counter()
        dataframe = pd.DataFrame(
            {'X': T[:, 0], 'Y': T[:, 1], 'Dimensionality of the target projection space': Dimension[j],
             'CPU Time': (endtime - starttime)})
        dataframe.to_csv(f'..\\Preliminary_Result\\RP_PCA_with_D_{Dimension[j]}.csv', index=False, sep=',')
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
        plt.savefig(f'..\\Preliminary_Result\\RP_PCA_with_D_{Dimension[j]}.png', dpi=600)
elif LOOP == 2:
    for j in range(0, len(Dimension)):
        projector = random_projection.GaussianRandomProjection(n_components=int(Dimension[j]))
        SNP_RP = projector.fit_transform(SNP_Raw)
        starttime = time.perf_counter()
        nmds = manifold.MDS(n_components=2, normalized_stress="auto").fit(SNP_RP)
        endtime = time.perf_counter()
        T = nmds.embedding_.copy()
        dataframe = pd.DataFrame({'X': T[:, 0], 'Y': T[:, 1], 'CPU Time': (endtime - starttime)})
        dataframe.to_csv(f'..\\Preliminary_Result\\RP_MDS_with_D_{Dimension[j]}.csv', index=False, sep=',')
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
        plt.savefig(f'..\\Preliminary_Result\\RP_MDS_with_D_{Dimension[j]}.png', dpi=600)
else:
    raise ValueError("LOOP has to be 1 or 2")
