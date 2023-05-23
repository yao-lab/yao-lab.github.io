import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from scipy import spatial
from scipy.stats import kendalltau

# load the data
data = sio.loadmat("face.mat")
faces = data["Y"]
id = data["id"]
##########################################################################################
# compute the pairwise distance matrix
dist_mat = np.zeros((33, 33))
for i in range(33):
    for j in range(i + 1, 33):
        dist = np.linalg.norm(faces[:, :, i] - faces[:, :, j])
        dist_mat[i, j] = dist
        dist_mat[j, i] = dist

# compute the affinity matrix using the Gaussian kernel
sigma = np.median(dist_mat)
aff_mat = np.exp(-(dist_mat**2) / (2 * sigma**2))

# compute the normalized affinity matrix
D = np.diag(np.sum(aff_mat, axis=1))
L = D - aff_mat
norm_aff_mat = np.dot(np.linalg.inv(D), np.dot(L, np.linalg.inv(D)))

# compute the transition matrix using the Markov Chain approach
trans_mat = np.dot(np.linalg.inv(D), aff_mat)

# compute the eigenvalues and eigenvectors of the transition matrix
eigvals, eigvecs = np.linalg.eig(trans_mat)
eigvecs = eigvecs[:, eigvals.argsort()]  # sort eigenvectors by eigenvalues
# extract the second smallest eigenvector
diffusion_embedding = eigvecs[:, 1]
##########################################################################################
# compute the pairwise distance matrix
distances = np.zeros((33, 33))
for i in range(33):
    for j in range(i + 1, 33):
        distances[i, j] = np.linalg.norm(faces[..., i].ravel() - faces[..., j].ravel())
        distances[j, i] = distances[i, j]

# Compute the dissimilarities from the distances
dissimilarities = distances.max() - distances
# Compute the MDS embedding on the top two eigenvectors
mds = MDS(n_components=2, random_state=0, dissimilarity="precomputed")
mds_embedding = mds.fit_transform(dissimilarities)[:, 0]
##########################################################################################
# create an Isomap object with n_components=1 and n_neighbors=5
isomap = Isomap(n_components=1, n_neighbors=5)
# fit the Isomap object to the data
X = faces.transpose(2, 0, 1).reshape((faces.shape[2], -1))
isomap.fit(X)
# get the embedded data
isomap_embedding = isomap.transform(X)[:, 0]
##########################################################################################
# create a LocallyLinearEmbedding object with n_components=1, n_neighbors=5, and method='standard'
lle = LocallyLinearEmbedding(n_components=1, n_neighbors=5, method="standard")
# fit the LLE object to the data
X = faces.transpose(2, 0, 1).reshape((faces.shape[2], -1))
lle.fit(X)
# get the embedded data
lle_embedding = lle.transform(X)[:, 0]
##########################################################################################
# create a LocallyLinearEmbedding object with n_components=1, n_neighbors=5, and method='ltsa'
lle = LocallyLinearEmbedding(n_components=1, n_neighbors=5, method="ltsa")
# fit the LLE object to the data
X = faces.transpose(2, 0, 1).reshape((faces.shape[2], -1))
lle.fit(X)
# get the embedded data
ltsa_embedding = lle.transform(X)[:, 0]
##########################################################################################
# Use t-SNE to reduce the dimensionality of the face data to 2D
tsne = TSNE(n_components=1)
X = faces.transpose(2, 0, 1).reshape((faces.shape[2], -1))
tsne_embedding = tsne.fit_transform(X)[:, 0]
##########################################################################################
# Use UMAP to reduce the dimensionality of the face data to 2D
model = umap.UMAP(n_components=1, n_neighbors=5)
# Fit model to data
X = faces.transpose(2, 0, 1).reshape((faces.shape[2], -1))
umap_embedding = model.fit_transform(X)[:, 0]
##########################################################################################

# create a figure and axis object
plt.figure(figsize=(10, 6))

# Plot the first eigenvector with Seaborn
ax1 = plt.subplot(811)
sns.heatmap(id / np.max(id), cmap="coolwarm", cbar=False, ax=ax1)
ax1.set_title("Ground Truth", fontsize=14)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
ax1 = plt.subplot(812)
sns.heatmap(diffusion_embedding.reshape((1, -1)) / np.max(diffusion_embedding), cmap="coolwarm", cbar=False, ax=ax1)
ax1.set_title("Diffusion Map", fontsize=14)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
ax1 = plt.subplot(813)
sns.heatmap(mds_embedding.reshape((1, -1)) / np.max(mds_embedding), cmap="coolwarm", cbar=False, ax=ax1)
ax1.set_title("MDS", fontsize=14)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
ax1 = plt.subplot(814)
sns.heatmap(isomap_embedding.reshape((1, -1)) / np.max(isomap_embedding), cmap="coolwarm", cbar=False, ax=ax1)
ax1.set_title("ISOMAP", fontsize=14)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
ax1 = plt.subplot(815)
sns.heatmap(lle_embedding.reshape((1, -1)) / np.max(lle_embedding), cmap="coolwarm", cbar=False, ax=ax1)
ax1.set_title("LLE", fontsize=14)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
ax1 = plt.subplot(816)
sns.heatmap(ltsa_embedding.reshape((1, -1)) / np.max(ltsa_embedding), cmap="coolwarm", cbar=False, ax=ax1)
ax1.set_title("LTSA", fontsize=14)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
ax1 = plt.subplot(817)
sns.heatmap(tsne_embedding.reshape((1, -1)) / np.max(tsne_embedding), cmap="coolwarm", cbar=False, ax=ax1)
ax1.set_title("TSNE", fontsize=14)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
ax1 = plt.subplot(818)
sns.heatmap(umap_embedding.reshape((1, -1)) / np.max(umap_embedding), cmap="coolwarm", cbar=False, ax=ax1)
ax1.set_title("UMAP", fontsize=14)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
#######################################################################################
# similarity
sorted_indices = np.argsort(diffusion_embedding)
tau, p_value = kendalltau(id[0, sorted_indices], range(1, 34))
print("diffusion", tau)
sorted_indices = np.argsort(mds_embedding)
tau, p_value = kendalltau(id[0, sorted_indices], range(1, 34))
print("MDS", tau)
sorted_indices = np.argsort(isomap_embedding)
tau, p_value = kendalltau(id[0, sorted_indices], range(1, 34))
print("ISOMAP", tau)
sorted_indices = np.argsort(lle_embedding)
tau, p_value = kendalltau(id[0, sorted_indices], range(1, 34))
print("LLE", tau)
sorted_indices = np.argsort(ltsa_embedding)
tau, p_value = kendalltau(id[0, sorted_indices], range(1, 34))
print("LTSA", tau)
sorted_indices = np.argsort(tsne_embedding)
tau, p_value = kendalltau(id[0, sorted_indices], range(1, 34))
print("TSNE", tau)
sorted_indices = np.argsort(umap_embedding)
tau, p_value = kendalltau(id[0, sorted_indices], range(1, 34))
print("UMAP", tau)
