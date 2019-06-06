import torch
import torch.utils.data
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.manifold import MDS

def normalize(x):
    mean = 0.1307
    std = 0.3081
    x = (x/255-mean)/std
    x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])
    return x

num_used = 2000

scale = np.array([1 / 0.3081])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./mnist', train=True))

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./mnist', train=False))

X_train = normalize(train_loader.dataset.train_data.numpy())[:num_used,:]
Y_train = train_loader.dataset.train_labels.numpy()[:num_used]
X_test = normalize(test_loader.dataset.test_data.numpy())[:num_used,:]
Y_test = test_loader.dataset.test_labels.numpy()[:num_used]

transformer = MDS(n_components = 2, max_iter=100, n_init=1)
X_transformed = transformer.fit_transform(X_train)

f = plt.figure()
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=Y_train, s=8)
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.title('MDS-2D')
f.savefig('result/MDS_2D.jpg',dpi=200)




