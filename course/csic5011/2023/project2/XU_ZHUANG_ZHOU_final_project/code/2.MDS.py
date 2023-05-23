import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS

# load the data
data = sio.loadmat("face.mat")
faces = data["Y"]
id = data["id"]
# compute the pairwise distance matrix
distances = np.zeros((33, 33))
for i in range(33):
    for j in range(i + 1, 33):
        distances[i, j] = np.linalg.norm(faces[..., i].ravel() - faces[..., j].ravel())
        distances[j, i] = distances[i, j]

# Compute the dissimilarities from the distances
dissimilarities = distances.max() - distances
print(dissimilarities)

# Compute the MDS embedding on the top two eigenvectors
mds = MDS(n_components=2, random_state=0, dissimilarity="precomputed")
embedding = mds.fit_transform(dissimilarities)
# plot the first eigenvector
plt.figure(figsize=(8, 2))
sns.heatmap(embedding[:, 0].reshape((1, -1)), cmap="coolwarm", cbar=False)
plt.title("The top 1st Eigenvector of MDS embedding", fontsize=16)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
# sort the faces by the value of the top 1st eigenvector of MDS embedding
sorted_indices = np.argsort(embedding[:, 0])
sorted_id = id[:, sorted_indices]
sorted_faces = faces[:, :, sorted_indices]
# create a figure and plot the images (random ranking)
fig, axs = plt.subplots(3, 11, figsize=(15, 5))
for i, ax in enumerate(axs.flat):
    ax.imshow(sorted_faces[:, :, i], cmap="gray")
    ax.set_axis_off()
    ax.set_title("ID {}".format(sorted_id[0, i]))
# display the figure
plt.show()
