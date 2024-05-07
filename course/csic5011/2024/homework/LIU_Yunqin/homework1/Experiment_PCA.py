import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# (a) Set up data matrix X
raw_data = pd.read_csv('train.7.txt')
data = np.array(raw_data,dtype='float32')
data = np.transpose(data)
data.shape

# (b) Compute the sample mean
mu = np.mean(data, axis=1)
X_tidle = data - mu[:,None]

# (c) Compute top k SVD
u, s, vh = np.linalg.svd(X_tidle)

# (d) Plot eigenvalue curve
plt.plot(s/s.sum())
plt.show()

# (e)Use imshow to visualize the mean and top-k principle components
k = 15
fig, axs = plt.subplots(int(np.ceil((k+1)/4)), 4)
axs[0,0].imshow(np.reshape(mu,(16,16)),cmap='gray')
axs[0,0].set_title('mu')
for i in range(k):
    axs[int(np.floor((i+1)/4)), (i+1)%4].imshow(np.reshape(u[:,i],(16,16)), cmap='gray')
    axs[int(np.floor((i+1)/4)), (i+1)%4].set_title(f'u_{i}')
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=1.,
                    top=1.,
                    wspace=0.4,
                    hspace=0.8)
plt.show()

# (f)
np.argsort(vh[0])
fig = plt.figure(figsize=(12, 5), dpi=80)
subfigs = fig.subfigures(1, 2)
ax_left = subfigs[0].subplots()
ax_right = subfigs[1].subplots(3,6)
ax_left.scatter(vh[0], vh[1], color="blue", s=4)
ax_left.set_xlabel("First Principal Component")
ax_left.set_ylabel("Second Principal Component")
ax_left.grid(True)
for second in np.arange(3):
    for first in np.arange(6):
        distance_to_grid = (vh[0]-(first*0.02-0.04))**2 + (vh[1]-(-second*0.05+0.05))**2
        min_index = np.argmin(distance_to_grid)
        ax_left.scatter(vh[0,min_index], vh[1,min_index], s=50, facecolors='none', edgecolors='r')
        ax_right[second,first].imshow(np.reshape(data[:,min_index],(16,16)), cmap='gray')
        ax_right[second,first].set_xticks([])
        ax_right[second,first].set_yticks([])
plt.show()