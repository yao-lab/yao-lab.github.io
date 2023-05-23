import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

# load the data
data = sio.loadmat("face.mat")
faces = data["Y"]
id = data["id"]
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
psi = eigvecs[:, 1]
# display this eigenvector
plt.figure(figsize=(8, 2))
sns.heatmap(psi.reshape((1, -1)), cmap="coolwarm", cbar=False)
plt.title("Second Smallest Eigenvector of Markov Chain", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks([])
plt.tight_layout()
plt.show()
# sort the faces by the value of the second smallest eigenvector
sorted_indices = np.argsort(psi)
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
