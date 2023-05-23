import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import umap
import seaborn as sns

# load the data
data = sio.loadmat("face.mat")
faces = data["Y"]
id = data["id"]
# Use UMAP to reduce the dimensionality of the face data to 2D
model = umap.UMAP(n_components=1, n_neighbors=5)
# Fit model to data
X = faces.transpose(2, 0, 1).reshape((faces.shape[2], -1))
faces_umap = model.fit_transform(X)
# get the indices that sort the data according to the first dimension of the embedded data
sorted_indices = np.argsort(faces_umap[:, 0])
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
