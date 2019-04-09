# -*- coding: utf-8 -*-
"""
@author: Yunfei YANG
"""

import numpy as np
import matplotlib.pyplot as plt
import gzip
import matplotlib

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn import svm
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import confusion_matrix
#from sklearn.datasets import fetch_openml
#from sklearn.model_selection import train_test_split
#import pickle
#from PIL import Image

#Load data (MNIST)
#X, y = fetch_openml('mnist_784', version=1, cache=True, return_X_y=True)
#X = X/255
#X_train_o, X_test, y_train_o, y_test = train_test_split(X, y, test_size=10000, random_state=123)
#print(X_train_o.shape)
#print(y_train_o.shape)

#Load data

f=gzip.GzipFile('../exp/data/train.gz')
file_content = f.read()
data = np.fromstring(file_content, dtype=float,sep=' ')
data = data.reshape(-1,257)
X_train_o = data[:,1:]
y_train_o = data[:,0]
#print(X_train_o.shape)
#print(y_train_o.shape)

f=gzip.GzipFile('../exp/data/test.gz')
file_content = f.read()
data = np.fromstring(file_content, dtype=float,sep=' ')
data = data.reshape(-1,257)
X_test = data[:,1:]
y_test = data[:,0]
#print(X_test.shape)

#MDS
load_mds = True
if load_mds == False:
    embedding = MDS(n_components=2)
    Xmds = embedding.fit_transform(X_train_o)
    #print(Xmds.shape)

    np.savez('../exp/project1/mds.npz', Xmds = Xmds)
else:
    file = np.load('../exp/project1/mds.npz')
    Xmds = file['Xmds']

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
plt.figure(figsize=(8,8))
plt.axis('off')
plt.scatter(Xmds[:,0], Xmds[:,1], c=y_train_o,cmap=matplotlib.colors.ListedColormap(colors))
plt.savefig('../exp/project1/mds.eps')
plt.show()


#classification
train_size = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7291]
var = [0.5,0.7,0.9]


#no pca
acc=[]
for s in train_size:
        
    X_train = X_train_o[0:s]
    y_train = y_train_o[0:s]
    
        
    #classifier
    #svm
    #classifier = svm.SVC(C=5,gamma=0.05)
    classifier = svm.SVC(gamma='auto')
    
    #logist
    #classifier = LogisticRegression(random_state=123,solver='newton-cg', multi_class='multinomial')
    
    classifier.fit(X_train, y_train)
    acc.append(classifier.score(X_test,y_test))
    #print(confusion_matrix(y_test, classifier.predict(X_test)))

np.savez('../exp/project1/svm_acc.npz', train_size=train_size, acc=acc)
#np.savez('../exp/project1/log_acc.npz', train_size=train_size, acc=acc)
#
#


#pca
for p in var:
    acc = []
    comp = []
    for s in train_size:
        
        X_train = X_train_o[0:s]
        y_train = y_train_o[0:s]
        
        pca = PCA(n_components = p)
        pca.fit(X_train)
        
        #classifier
        #svm
        classifier = svm.SVC(gamma='auto')
        
        #logist
        #classifier = LogisticRegression(random_state=123,solver='newton-cg', multi_class='multinomial')
        
        X_train = pca.transform(X_train)
        comp.append(np.size(X_train,1))
        
        classifier.fit(X_train, y_train)
        acc.append(classifier.score(pca.transform(X_test),y_test))
        
        # save the model
#        filename = '../exp/project1/pca'+str(int(10*p)) +'svm_'+str(s)+'.sav'
#        pickle.dump(classifier, open(filename, 'wb'))

    np.savez('../exp/project1/pca'+str(int(10*p))+'_svm_acc.npz', train_size=train_size, comp=comp, acc=acc)
#    np.savez('../exp/project1/pca'+str(int(10*p))+'_log_acc.npz', train_size=train_size, comp=comp, acc=acc)



# load model
#loaded_model = pickle.load(open(filename, 'rb'))



