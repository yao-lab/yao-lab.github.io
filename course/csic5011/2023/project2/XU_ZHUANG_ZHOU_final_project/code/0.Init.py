import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# load the data
data = sio.loadmat("face.mat")
faces = data["Y"]
id = data["id"]
#
# random ranking
#
# create a figure and plot the images (random ranking)
fig, axs = plt.subplots(3, 11, figsize=(15, 5))
for i, ax in enumerate(axs.flat):
    ax.imshow(faces[:, :, i], cmap="gray")
    ax.set_axis_off()
    ax.set_title("ID {}".format(id[0, i]))
# display the figure
plt.show()
#
# real ranking
#
# sort the ID array and get the corresponding indices
sorted_indices = np.argsort(id[0, :])
sorted_id = id[:, sorted_indices]
# rearrange the face array based on the sorted ID array
sorted_faces = faces[:, :, sorted_indices]
# create a figure and plot the images (random ranking)
fig, axs = plt.subplots(3, 11, figsize=(15, 5))
for i, ax in enumerate(axs.flat):
    ax.imshow(sorted_faces[:, :, i], cmap="gray")
    ax.set_axis_off()
    ax.set_title("ID {}".format(sorted_id[0, i]))
# display the figure
plt.show()
