import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import Isomap

# load the data
data = sio.loadmat("face.mat")
faces = data["Y"]
id = data["id"]
# create an Isomap object with n_components=1 and n_neighbors=5
isomap = Isomap(n_components=1, n_neighbors=5)
# fit the Isomap object to the data
X = faces.transpose(2, 0, 1).reshape((faces.shape[2], -1))
isomap.fit(X)
# get the embedded data
faces_isomap = isomap.transform(X)
# plot
plt.figure(figsize=(8, 2))
sns.heatmap(faces_isomap[:, 0].reshape((1, -1)), cmap="coolwarm", cbar=False)
plt.title("ISOMAP embedding", fontsize=16)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
# get the indices that sort the data according to the first dimension of the embedded data
sorted_indices = np.argsort(faces_isomap[:, 0])
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
