from time import time
import numpy as np
import numpy.linalg as alg
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import offsetbox
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding

X = np.loadtxt('face.csv', delimiter=',')
X = X.T

(N, P)= X.shape
print('The shape of face matrix now is {}'.format(X.shape))

# Normalization
one = np.ones((N,1))
X = X-one.dot(one.T).dot(X)
u, s, vh = alg.svd(X)

print('The shape of u is', u.shape)
print('The shape of s is', s.shape)
print('The shape of v is', vh.shape)

M = u[:,:2]
p1 = M[:, 0]
p2 = M[:, 1]

pairs = [(p1[i], i) for i in range(N)]
pairs.sort(key = lambda pairs: pairs[0])

order = [j for (i, j) in pairs]

plt.figure()
cmap = cm.gray_r
X = np.loadtxt('face.csv', delimiter=',')

for i in range(N):
    idx = order[i]
    pic = X[:, idx]
    pic_matrix = np.reshape(pic, (92, 112))

    plt.subplot(3, 11, i + 1)
    plt.imshow(pic_matrix.T, cmap=cmap)
    plt.xticks([])
    plt.yticks([])

plt.show()

Original_X = X
X = X.T
print(X.shape)


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    plt.scatter(X[:, 0], X[:, 1])

    if hasattr(offsetbox, 'AnnotationBbox'):
        for i in range(X.shape[0]):
            pic = Original_X[:, i]
            pic_matrix = np.reshape(pic, (92, 112))
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(pic_matrix.T, cmap=cmap, zoom=.2),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

    print('MDS')
    t0 = time()
    mds = MDS(n_components=2, n_init=1, max_iter=100)
    X_mds = mds.fit_transform(X)
    plot_embedding(X_mds, 'MDS Projection of 33 faces (time %.2f)' % (time() - t0))

    print("Isomap")
    t0 = time()
    iso = Isomap(n_neighbors=5, n_components=2)
    X_iso = iso.fit_transform(X)
    plot_embedding(X_iso, 'Isomap Projection of 33 faces (time %.2f)' % (time() - t0))

    print('LLE')
    t0 = time()
    lle = LocallyLinearEmbedding(n_neighbors=5, n_components=2)
    X_lle = lle.fit_transform(X)
    plot_embedding(X_lle, 'LLE Projection of 33 faces (time %.2f)' % (time() - t0))