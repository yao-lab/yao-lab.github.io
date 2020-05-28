# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:09:50 2020

@author: zchenef
"""

import cv2
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import time

from pydiffmap import diffusion_map as dm   # need to install pydiffmap package
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import TSNE

mat = io.loadmat('face.mat')
name=np.transpose(mat['id']).squeeze()
img=np.array(mat['Y']).transpose(2,0,1)

###plot the 33 image
for i in range(33):
    # plt.imshow(img[i],cmap='gray')
    # plt.axis('off')
    # plt.savefig('image/{}.jpg'.format(name[i]))
    cv2.imwrite('image/{}.jpg'.format(name[i]),img[i])

###flatten the data
Y=img.reshape(33,-1)
label=np.array([9,13,19,32,6,18,28,7,17,1,5,16,12,10,4,21,22,26,33,11,2,24,3,27,29,23,14,30,31,20,15,25,8])

###Diffusion map
start=time.time()
mydmap = dm.DiffusionMap.from_sklearn(n_evecs = 2, alpha =1, epsilon = 'bgh', k=64)
Y1 = mydmap.fit_transform(Y)
end=time.time()
print('Diffusion map: {:.4f} sec'.format(end-start))

plt.scatter(Y1[:, 0], Y1[:, 1])
plt.title("Diffusion map" )
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#sort the diffusion map by the first eigenvecor and get the rank
sort1=np.argsort(Y1[:, 0])
rank1=sort1+1
E1=np.abs(np.argsort(sort1)+1-label).sum()

#plot the sorted image
for i in range(33):
    cv2.imwrite('diffusion map/{}.jpg'.format(i),img[sort1[i]])


###MDS
start=time.time()
mds = MDS(n_components=2)
Y2=mds.fit_transform(Y)
end=time.time()
print('MDS: {:.4f} sec'.format(end-start))

plt.scatter(Y2[:, 0], Y2[:, 1])
plt.title("MDS" )
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#sort MDS by the first eigenvecor and get the rank
sort2=np.argsort(-Y2[:, 0])
sort2_r=np.argsort(Y2[:, 0])
rank2=sort2+1
E2=np.abs(np.argsort(sort2)+1-label).sum()

#plot the sorted image
for i in range(33):
    cv2.imwrite('mds/{}.jpg'.format(i),img[sort2_r[i]])

###Isomap
start=time.time()
isomap= Isomap(n_neighbors=5,n_components=2)
Y3 = isomap.fit_transform(Y)
end=time.time()
print('Isomap: {:.4f} sec'.format(end-start))

plt.scatter(Y3[:, 0], Y3[:, 1])
plt.title("ISOMAP")
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#sort Isomap by the first eigenvecor and get the rank
sort3=np.argsort(Y3[:, 0])
rank3=sort3+1
E3=np.abs(np.argsort(sort3)+1-label).sum()

#plot the sorted image
for i in range(33):
    cv2.imwrite('isomap/{}.jpg'.format(i),img[sort3[i]])


###LLE
start=time.time()
lle= LocallyLinearEmbedding(n_neighbors=5,n_components=2)
Y4 = lle.fit_transform(Y)
end=time.time()
print('LLE: {:.4f} sec'.format(end-start))

plt.scatter(Y4[:, 0], Y4[:, 1])
plt.title("LLE")
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#sort LLE by the first eigenvecor and get the rank
sort4=np.argsort(-Y4[:, 0])
sort4_r=np.argsort(Y4[:, 0])
rank4=sort4+1
E4=np.abs(np.argsort(sort4)+1-label).sum()

#plot the sorted image
for i in range(33):
    cv2.imwrite('lle/{}.jpg'.format(i),img[sort4_r[i]])


###LSTA
start=time.time()
lsta= LocallyLinearEmbedding(n_neighbors=5,n_components=2,method = 'ltsa')
Y5 = lsta.fit_transform(Y)
end=time.time()
print('LSTA: {:.4f} sec'.format(end-start))

plt.scatter(Y5[:, 0], Y5[:, 1])
plt.title("LSTA")
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#sort LSTA by the first eigenvecor and get the rank
sort5=np.argsort(Y5[:, 0])
rank5=sort5+1
E5=np.abs(np.argsort(sort5)+1-label).sum()

#plot the sorted image
for i in range(33):
    cv2.imwrite('lsta/{}.jpg'.format(i),img[sort5[i]])

###TSNE
start=time.time()
tsne= TSNE(n_components=2)
Y6 = tsne.fit_transform(Y)
end=time.time()
print('TSNE: {:.4f} sec'.format(end-start))

plt.scatter(Y6[:, 0], Y6[:, 1])
plt.title("TSNE")
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#sort TSNE by the first eigenvecor and get the rank
sort6=np.argsort(Y6[:, 0])
rank6=np.argsort(sort6)+1
E6=np.abs(np.argsort(sort6)+1-label).sum()

#plot the sorted image
for i in range(33):
    cv2.imwrite('tsne/{}.jpg'.format(i),img[sort6[i]])



### different k
k=[i for i in range(2,10)]

r1=[]
for i in k:    
    m= Isomap(n_neighbors=i,n_components=2)
    x=m.fit_transform(Y)
    #adjust face order from left to right
    if(i==3 or i==8 or i==9):
        sort=np.argsort(-x[:, 0])
    else:
        sort=np.argsort(x[:, 0])
    # sort=np.argsort(x[:, 0])
    for j in range(33):
        cv2.imwrite('test/{}/{}.jpg'.format(i,j),img[sort[j]])

    E=np.abs(np.argsort(sort)+1-label).sum()
    r1.append(E)
    
r2=[]
for i in k: 
    m= LocallyLinearEmbedding(n_neighbors=5,n_components=2)
    x=m.fit_transform(Y)
    #adjust face order from left to right
    sort=np.argsort(-x[:, 0])
    for j in range(33):
        cv2.imwrite('test/{}/{}.jpg'.format(i,j),img[sort[j]])
    E=np.abs(np.argsort(sort)+1-label).sum()
    r2.append(E)
    
r3=[]
for i in k: 
    m= LocallyLinearEmbedding(n_neighbors=i,n_components=2,method = 'ltsa')
    x=m.fit_transform(Y)
    #adjust face order from left to right
    if(i==3 or i==5 or i==8):
        sort=np.argsort(x[:, 0])
    else:
        sort=np.argsort(-x[:, 0])
    # sort=np.argsort(x[:, 0])
    for j in range(33):
        cv2.imwrite('test/{}/{}.jpg'.format(i,j),img[sort[j]])
    E=np.abs(np.argsort(sort)+1-label).sum()
    r3.append(E)


plt.plot(k,r1)
plt.plot(k,r2)
plt.plot(k,r3)
plt.ylabel('Absolute error')             
plt.xlabel('The number of nearest neighbor')
plt.legend(['ISOMAP','LLE','LSTA'],bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)








