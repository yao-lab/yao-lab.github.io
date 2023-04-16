import numpy as np
import pandas as pd
from sklearn import random_projection

# 1- data loading ---- 
df_raw = pd.read_csv("data/ceph_hgdp_minor_code_XNA.betterAnnotated.csv")
df_raw.index = df_raw.loc[:, 'snp']
df_SNPs = df_raw.iloc[:, 3:1046]
mt_SNPs = df_SNPs.T

from sklearn.random_projection import johnson_lindenstrauss_min_dim
johnson_lindenstrauss_min_dim(n_samples=mt_SNPs.shape[0], eps=0.1)



# random projection with eps = 0.1 
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(mt_SNPs)
X_new.shape
df_DR = pd.DataFrame(X_new)
df_DR.index = mt_SNPs.index
df_DR.to_csv("results/RandProj_DR_results.csv", index=True)