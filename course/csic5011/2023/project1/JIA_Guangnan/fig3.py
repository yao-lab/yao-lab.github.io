#To visualize the PCA components as images for components 1 to 30, 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = "https://hastie.su.domains/ElemStatLearn/datasets/zip.train.gz"
data = pd.read_csv(url, sep=" ", header=None)

# Preprocess the dataset
X = data.iloc[:, 1:-1].values
y = data.iloc[:, 0].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with n_components=30
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA components as images for components 1 to 30
fig, axes = plt.subplots(6, 5, figsize=(12, 14))
for i, ax in enumerate(axes.flat):
    component = pca.components_[i].reshape(16, 16)
    ax.imshow(component, cmap='gray')
    ax.set_title(f"Component {i + 1}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('com1_30.png',pdi=500)
plt.show()