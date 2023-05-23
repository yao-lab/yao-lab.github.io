import numpy as np
import scipy.io as sio
from node2vec import Node2Vec
from sklearn.cluster import KMeans
import networkx as nx

mat_data = sio.loadmat('karate.mat')
A = mat_data['A']
label = mat_data['c0']

G = nx.DiGraph()
G.add_nodes_from(range(A.shape[0]))
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if A[i, j] == 1:
            G.add_edge(i, j)

node2vec = Node2Vec(G)
model = node2vec.fit()

matrix = np.zeros((len(model.wv.key_to_index), model.wv.vector_size))
for i, word in enumerate(model.wv.key_to_index):
    matrix[int(word)] = model.wv[word]

kmeans = KMeans(n_clusters=2)
kmeans.fit(matrix)
labels = kmeans.predict(matrix)



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

G = nx.DiGraph()
G.add_nodes_from(range(A.shape[0]))
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if A[i, j] == 1:
            G.add_edge(i, j)

node2vec = Node2Vec(G)
model = node2vec.fit()

matrix = np.zeros((len(model.wv.key_to_index), model.wv.vector_size))
for i, word in enumerate(model.wv.key_to_index):
    matrix[int(word)] = model.wv[word]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(matrix)
labels = kmeans.predict(matrix)
