import time
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA

SNP_Raw = np.load('SNP_Raw.npy').transpose()
Nation_Raw = np.load('Nation_Raw.npy', allow_pickle=True)

starttime = time.perf_counter()
nmds = manifold.MDS(n_components=2, normalized_stress="auto").fit(SNP_Raw)
endtime = time.perf_counter()
embedding = nmds.embedding_.copy()
dataframe = pd.DataFrame({'X': embedding[:, 0], 'Y': embedding[:, 1],  'CPU Time': (endtime - starttime)})
dataframe.to_csv('..\\Preliminary_Result\\Direct_MDS_Embedding.csv', index=False, sep=',')

pca = PCA(n_components=2)
embedding = pca.fit_transform(embedding)

Nations = list(set(Nation_Raw))
plt.figure()
lw = 1.5
colors = ['blue', 'orange', 'green', 'red', 'brown', 'pink', 'olive']
legend_used = []
for i in range(0, embedding.shape[0]):
    i_nation = Nation_Raw[i]
    i_color = colors[Nations.index(i_nation)]
    if i_nation in legend_used:
        plt.scatter(embedding[i, 0], embedding[i, 1], color=i_color, alpha=0.8, lw=lw)
    else:
        plt.scatter(embedding[i, 0], embedding[i, 1], color=i_color, alpha=0.8, lw=lw, label=i_nation)
        legend_used.append(i_nation)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
plt.axis('scaled')
plt.savefig('..\\Preliminary_Result\\Direct_MDS_followed_by_PCA_Embedding.png', dpi=600, bbox_inches='tight')
plt.show()
