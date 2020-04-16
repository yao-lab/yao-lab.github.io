# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:15:03 2020

@author: Z
"""
import numpy as np
from sklear import manifold
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
#data input
import gzip
read_file = gzip.GzipFile("zip.train.gz")
file = read_file.readlines()
data = np.zeros(shape = (len(file), 257))
for i in range(0, len(file)):
    data[i,:] = np.fromstring(file[i], dtype=float, sep=" ")
train_image = data[:,1:]
train_label = data[:,0]

read_file = gzip.GzipFile("zip.test.gz")
file = read_file.readlines()
data = np.zeros(shape = (len(file), 257))
for i in range(0, len(file)):
    data[i,:] = np.fromstring(file[i], dtype=float, sep=" ")
test_image = data[:,1:]
test_label = data[:,0]
x_all=np.concatenate((train_image,test_image),axis=0)


#######original data classification
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(train_image, train_label)
predict = rfc.predict(test_image)
print("accuracy_score: %.4lf" % accuracy_score(predict, test_label))
print("Classification report for classifier %s:\n%s\n" % (rfc, classification_report(test_label, predict)))

#######PFA classification
class PFA(object):
    def __init__(self, n_features, q):
        self.q = q
        self.n_features = n_features

    def fit(self, X):
        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA(n_components=self.q).fit(X)
        A_q = pca.components_.T

        kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]

        
q=train_image.shape[1]        
pfa = PFA(n_features=230,q=q)
pfa.fit(x_all)
# To get the transformed matrix
new_train = pfa.features_[:7291]
new_test = pfa.features_[7291:]

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(new_train, train_label)
predict = rfc.predict(new_test)
print("accuracy_score: %.4lf" % accuracy_score(predict, test_label))
print("Classification report for classifier %s:\n%s\n" % (rfc, classification_report(test_label, predict)))
##############PCA parallel analysis
X=train_image
n, dim = X.shape
mean = np.mean(X, axis = 0)
X0 = X - mean
#X0 = X
Cov = np.dot(X0.T, X0)/(n-1)


n_perm = 9

perc = 0.05

evals, evecs = np.linalg.eigh(Cov)  #original egenvalues and egenvectos
evals = evals[::-1]
evecs = evecs[:, ::-1]

Xcp = X0.copy()
evals_perm = np.zeros([n_perm, dim])  #9*256

for i in range(n_perm):
    for j in range(1, dim):
        np.random.shuffle(Xcp[:, j])
    Cov_perm = np.dot(Xcp.T, Xcp)/(n-1)
    evals_perm[i] = np.linalg.eigvalsh(Cov_perm)[::-1]

evals0 = np.mean(evals_perm, axis = 0)  #256
evals_perm = np.sort(evals_perm, axis = 0)[::-1]
evals_perc = evals_perm[int(np.floor(perc * n_perm))]   #256
pvals = np.mean((evals_perm > evals).astype(float), axis = 0)   #mean on True or False

# index of the first nonzero p-values
for j in range(dim):
    if pvals[j] > 0:
        pv1 = j
        break

new_evecs=evecs[:pv1,:]
pa_x=np.dot(Xcp,new_evecs.T)
#remaining num of values: pv1        

from sklearn.decomposition import PCA
pca = PCA(n_components=26)
x_pca = pca.fit_transform(x_all)
train_pca=x_pca[:7291]
test_pca=x_pca[7291:]

rfc.fit(train_pca, train_label)
predict = rfc.predict(test_pca)
print("accuracy_score: %.4lf" % accuracy_score(predict, test_label))
print("Classification report for classifier %s:\n%s\n" % (rfc, classification_report(test_label, predict)))

#########tsne classification
tsne = manifold.TSNE(n_components=26, init='pca', random_state=0)
x_t=tsne.fit_transform(x_all)
train_t=x_t[:7291]
test_t=x_t[7291:]

rfc.fit(train_t, train_label)

predict = gcv.predict(test_t)
print("accuracy_score: %.4lf" % accuracy_score(predict, test_label))
print("Classification report for classifier %s:\n%s\n" % (rfc, classification_report(test_label, predict)))








