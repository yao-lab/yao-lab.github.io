import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
url = "https://hastie.su.domains/ElemStatLearn/datasets/zip.train.gz"
data = pd.read_csv(url, sep=" ", header=None)

# Preprocess the dataset
X = data.iloc[:, 1:-1].values
y = data.iloc[:, 0].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Image Samples of the given dataset 
fig1, axes = plt.subplots(1, 4, figsize=(2 * 4, 2 * 1))
for i in range(4):

    ax = axes[i]
    ax.imshow(X[i,:].reshape(16,16), cmap='gray')
    ax.axis('off')
plt.show()


