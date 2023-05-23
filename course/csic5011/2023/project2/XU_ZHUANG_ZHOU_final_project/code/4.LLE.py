import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import LocallyLinearEmbedding

# load the data
data = sio.loadmat("face.mat")
faces = data["Y"]
id = data["id"]
# create a LocallyLinearEmbedding object with n_components=1, n_neighbors=5, and method='standard'
lle = LocallyLinearEmbedding(n_components=1, n_neighbors=5, method="standard")
# fit the LLE object to the data
X = faces.transpose(2, 0, 1).reshape((faces.shape[2], -1))
lle.fit(X)
# get the embedded data
faces_lle = lle.transform(X)
# get the indices that sort the data according to the first dimension of the embedded data
sorted_indices = np.argsort(faces_lle[:, 0])
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
