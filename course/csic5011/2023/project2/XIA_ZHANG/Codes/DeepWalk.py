import os
import numpy as np
import scipy.io as sio
import torch
import dgl
from dgl.nn import DeepWalk
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import networkx as nx

mat_data = sio.loadmat('karate.mat')
A = mat_data['A']
label = mat_data['c0']

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

src_ids = []
dst_ids = []
for i in range(34):
    for j in range(34):
        if A[i, j] == 1:
            src_ids.append(i)
            dst_ids.append(j)
src_ids = torch.tensor(src_ids)
dst_ids = torch.tensor(dst_ids)
g = dgl.graph((src_ids, dst_ids))

model = DeepWalk(g)
model = model.to(device)

dataloader = DataLoader(torch.arange(g.num_nodes()), batch_size=156,
                        shuffle=True, collate_fn=model.sample)
optimizer = SparseAdam(model.parameters(), lr=0.01)
num_epochs = 50

for epoch in range(num_epochs):
    for batch_walk in dataloader:
        loss = model(batch_walk.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

kmeans = KMeans(n_clusters=2)
kmeans.fit(model.node_embed.weight.detach().cpu().numpy())
labels = kmeans.predict(model.node_embed.weight.detach().cpu().numpy())



G = nx.read_gml('polblogs.gml')
label = []
for node in G.nodes():
    label.append(G.nodes[node]['value'])
label = np.array(label)
A = nx.to_numpy_matrix(G)
A = A+np.transpose(A)
A[A > 1] = 1
deg = np.sum(A, axis=0)
place = np.where(deg == 0)[1]
pp = []
for i in range(1490):
    if i in place:
        continue
    pp.append(i)
A = A[np.ix_(pp, pp)]
src_ids = []
dst_ids = []
for i in range(1224):
    for j in range(1224):
        if A[i, j] == 1:
            src_ids.append(i)
            dst_ids.append(j)
src_ids = torch.tensor(src_ids)
dst_ids = torch.tensor(dst_ids)
g = dgl.graph((src_ids, dst_ids))
model = DeepWalk(g)
model = model.to(device)
dataloader = DataLoader(torch.arange(g.num_nodes()), batch_size=256,
                        shuffle=True, collate_fn=model.sample)
optimizer = SparseAdam(model.parameters(), lr=0.01)
num_epochs = 50
for epoch in range(num_epochs):
    for batch_walk in dataloader:
        loss = model(batch_walk.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
kmeans = KMeans(n_clusters=2)
kmeans.fit(model.node_embed.weight.detach().cpu().numpy())
labels = kmeans.predict(model.node_embed.weight.detach().cpu().numpy())
