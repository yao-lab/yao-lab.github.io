# %%
from scipy import io as sio
import numpy as np
from pydiffmap import diffusion_map
from matplotlib import pyplot as plt
from sklearn import manifold
# %%
data = sio.loadmat('face.mat')
raw_data = data['Y']
gt = np.asarray([9,13,19,32,6,18,28,7,17,1,5,16,12,10,4,21,22,26,33,11,2,24,3,27,29,23,14,30,31,20,15,25,8])
idx = np.argsort(gt)
print(idx + 1)
gt_mat = np.zeros((33, 33))
for i in range(33):
    gt_mat[i, idx[i]] = 1
plt.figure(figsize=[12, 5])
for i in range(33):
    plt.subplot(3, 11, i + 1)
    plt.imshow(raw_data[:, :, idx[i]], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
imgs = raw_data.reshape(-1, 33).transpose(1, 0).astype(np.float)
num_n = [4, 8, 16, 32]
# %%
rank_mat_1 = np.zeros_like(gt_mat)
rank_mat_2 = np.zeros_like(gt_mat)
min_error = np.inf
min_rank = None

dmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs=2, alpha=0.5, epsilon='bgh', k=32)
ev = dmap.fit_transform(imgs)
rank_1 = ev[:, 0].argsort()
rank_2 = rank_1[::-1]

for i in range(33):
    rank_mat_1[i, rank_1[i]] = 1
    rank_mat_2[i, rank_2[i]] = 1
error_1 = np.linalg.norm(rank_mat_1 - gt_mat)
error_2 = np.linalg.norm(rank_mat_2 - gt_mat)
print('error_1:', error_1)
print('error_2:', error_2)
if error_1 < min_error:
    min_error = error_1
    min_rank = rank_1
if error_2 < min_error:
    min_error = error_2
    min_rank = rank_2

print(min_rank + 1)
print(min_error)

plt.figure(figsize=[12, 5])
plt.scatter(ev[:, 0], ev[:, 1])
for i in range(33):
    plt.text(ev[i, 0] + 0.03, ev[i, 1] + 0.03, str(i + 1), fontsize=9)
plt.axis('off')
plt.title('Diffusion map')
plt.show()

plt.figure(figsize=[12, 5])
for i in range(33):
    plt.subplot(3, 11, i + 1)
    plt.imshow(raw_data[:, :, min_rank[i]], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
# %%
mds = manifold.MDS(n_components=2)
ev = mds.fit_transform(imgs)
rank = ev[:, 0].argsort()
print(ev[:, 0])
print(rank + 1)

rank_mat = np.zeros_like(gt_mat)
for i in range(33):
    rank_mat[i, rank[i]] = 1
error = np.linalg.norm(rank_mat - gt_mat)
print(error)

plt.figure(figsize=[12, 5])
plt.scatter(ev[:, 0], ev[:, 1])
for i in range(33):
    plt.text(ev[i, 0] + 0.03, ev[i, 1] + 0.03, str(i + 1), fontsize=9)
plt.axis('off')
plt.title('MDS')
plt.show()

plt.figure(figsize=[12, 5])
for i in range(33):
    plt.subplot(3, 11, i + 1)
    plt.imshow(raw_data[:, :, rank[i]], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=[12, 5])
for i in range(33):
    plt.subplot(3, 11, i + 1)
    plt.imshow(raw_data[:, :, i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
# %%
