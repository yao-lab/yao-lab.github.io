import time
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#===================================================================================#
LOOP = 2  # 1 for generating eigenvalues; 2 for generating score plot
#===================================================================================#

SNP_Raw = np.load('SNP_Raw.npy')
Nation_Raw = np.load('Nation_Raw.npy', allow_pickle=True)
starttime = time.perf_counter()
pca = PCA(svd_solver='full').fit(SNP_Raw.T)
endtime = time.perf_counter()

if LOOP == 1:
    dataframe = pd.DataFrame({'Sigular Value': pca.singular_values_, 'CPU Time': (endtime - starttime)})
    dataframe.to_csv('..\\Preliminary_Result\\Direct_PCA_EigenValue.csv', index=False, sep=',')
    x = 1 + np.arange(20)
    fig, ax = plt.subplots()
    ax.bar(x, pca.singular_values_[0:20], color='blue', edgecolor='black', linewidth=1.2)
    ax.set(xlim=(0.5, 20), xticks=np.arange(1, 21),
           ylim=(0, int(1.2 * pca.singular_values_.max())))
    plt.show()
    plt.savefig('..\\Preliminary_Result\\Direct_PCA_EigenValue.png', dpi=600)
elif LOOP == 2:
    Nations = list(set(Nation_Raw))
    T = pca.components_[0:2, :]
    X = T[0, :] @ SNP_Raw
    Y = T[1, :] @ SNP_Raw
    plt.figure()
    lw = 1.5
    colors = ['blue', 'orange', 'green', 'red', 'brown', 'pink', 'olive']
    legend_used = []
    for i in range(0, X.size):
        i_nation = Nation_Raw[i]
        i_color = colors[Nations.index(i_nation)]
        if i_nation in legend_used:
            plt.scatter(X[i], Y[i], color=i_color, alpha=0.8, lw=lw)
        else:
            plt.scatter(X[i], Y[i], color=i_color, alpha=0.8, lw=lw, label=i_nation)
            legend_used.append(i_nation)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.savefig('..\\Preliminary_Result\\Direct_PCA_2D_Scores.png', dpi=600)
    plt.show()
else:
       raise ValueError("LOOP has to be 1 or 2")