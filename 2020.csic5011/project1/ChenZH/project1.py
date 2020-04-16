# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:24:39 2020

@author: CHEN ZHENGHUI
"""

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA 
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns


mat = io.loadmat('snp452-data.mat')
X=np.transpose(mat['X'])
y=np.transpose(mat['stock'])

class_count={}
class_name=[]
for i in range(452):
    cla=y[i][0][0][0][2].tolist()[0]
    if  cla not in class_name:
        class_name.append(cla)
        class_count[cla]=1
    else:
        class_count[cla]+=1

###plot the distribution of stock class   
num_list = [ class_count[key] for key in class_count ]  
name_list = ['IND', 'FIN','HC','CD','IT','UT','MA','CS','TS','EN' ]  
plt.bar(range(len(num_list)), num_list,tick_label=name_list)  
plt.ylabel('Numbers of Different Class')             
plt.xlabel('Class of Stock')
plt.show()  

###plot the prcie of stocks by days
for i in range(452):
    plt.plot(range(X.shape[1]), X[i,:].squeeze()) #折线图
plt.ylabel('Price of Stock')             
plt.xlabel('Days')

###plot the growth rate of stock
x=X[:,1:1258]-X[:,0:1257]
rate=np.zeros((452,1257))
for i in range(452):
    for j in range(1257):
        rate[i][j]=x[i][j]/X[i][j]
for i in range(452):
    plt.plot(range(1257), rate[i,:].squeeze()) #折线图
plt.ylabel('Growth Rate of Stock')             
plt.xlabel('Days')
    
###class label
label={}
y_label=[]
for i, name in enumerate(class_name):
    label[name]=i
for i in range(452):
    cla=y[i][0][0][0][2].tolist()[0]
    y_label.append(label[cla])

### preprocessing
scaler=StandardScaler()
scaler.fit(rate)
rate_scaled=scaler.transform(rate)

sns.distplot(rate_scaled[0])
plt.xlim(-5, 5)
plt.ylim(0, 1)
#sum(rate_scaled[:,0])

###PCA
pca = PCA(n_components=0.95)
pca.fit(rate_scaled)
a=pca.explained_variance_ratio_
c=pca.n_components_
plt.scatter(range(c),a,s=20, alpha=0.6)
plt.xlim(-2, 200)
plt.ylim(-0.01, 0.08)
plt.ylabel('Explained_variance_ratio')             
plt.xlabel('PCA component')    
#b=[sum(a[:i+1]) for i in range(len(a))]
#plt.scatter(range(c),b)


pca = PCA(n_components=2)
pca.fit(rate_scaled)
newx=pca.transform(rate_scaled)
print(pca.explained_variance_ratio_)
arr=np.array(y_label)
for i in range(10):
    plt.scatter(newx[np.where(arr==i)][:,0],newx[np.where(arr==i)][:,1])
plt.ylabel('Second principal component')             
plt.xlabel('First principal component')    


###SparsePCA
transformer = SparsePCA(n_components=2, random_state=0)
transformer.fit(rate_scaled)
x_transformed = transformer.transform(rate_scaled)
arr=np.array(y_label)
for i in range(10):
    plt.scatter(x_transformed[np.where(arr==i)][:,0],newx[np.where(arr==i)][:,1])
plt.ylabel('Second principal component')             
plt.xlabel('First principal component')    


###MDS
embedding = MDS(n_components=2)
x_transformed = embedding.fit_transform(rate_scaled)
arr=np.array(y_label)
for i in range(10):
    plt.scatter(x_transformed[np.where(arr==i)][:,0],newx[np.where(arr==i)][:,1])
plt.ylabel('Second principal component')             
plt.xlabel('First principal component')    

###Isomap
embedding = Isomap(n_components=2)
x_transformed = embedding.fit_transform(rate_scaled)
arr=np.array(y_label)
for i in range(10):
    plt.scatter(x_transformed[np.where(arr==i)][:,0],newx[np.where(arr==i)][:,1])
plt.ylabel('Second principal component')             
plt.xlabel('First principal component')

###LLE
embedding = LocallyLinearEmbedding(n_components=2)
x_transformed = embedding.fit_transform(rate_scaled)
arr=np.array(y_label)
for i in range(10):
    plt.scatter(x_transformed[np.where(arr==i)][:,0],newx[np.where(arr==i)][:,1])
plt.ylabel('Second principal component')             
plt.xlabel('First principal component')

###TSNE
embedding = TSNE(n_components=2)
x_transformed = embedding.fit_transform(rate_scaled)
arr=np.array(y_label)
for i in range(10):
    plt.scatter(x_transformed[np.where(arr==i)][:,0],newx[np.where(arr==i)][:,1])
plt.ylabel('Second principal component')             
plt.xlabel('First principal component')      



###classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

###exclude TS class
arr=np.array(y_label)
index=np.where(arr==8)
X1=np.delete(rate_scaled,index[0],0)
y1=np.delete(arr,index[0])
x_train,x_test,y_train,y_test=train_test_split(X1,y1,test_size=0.3,random_state=0)

###class count
train_count={}
test_count={}
for i in y_train:
    if i not in train_count:
        train_count[i]=1
    else:
        train_count[i]+=1
for i in y_test:
    if i not in test_count:
        test_count[i]=1
    else:
        test_count[i]+=1    

###LogisticRegression    
model= LogisticRegression(solver='liblinear',random_state=0,multi_class='auto')
model.fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
test_predicted = model.predict(x_test)
cm = confusion_matrix(y_test, test_predicted)

###RandomForest
forest = RandomForestClassifier(n_estimators=300,random_state=0)
forest.fit(x_train, y_train)
print(forest.score(x_train, y_train))
print(forest.score(x_test, y_test))
train_predicted = forest.predict(x_train)
test_predicted = forest.predict(x_test)
cm = confusion_matrix(y_test, test_predicted)



###PCA+LR/RF
train_acc=[]
test_acc=[]
for i in range(2,200,10):
    x_train,x_test,y_train,y_test=train_test_split(X1,y1,test_size=0.3,random_state=0)
    pca = PCA(n_components=i)
    x_train=pca.fit_transform(x_train)
    x_test=pca.transform(x_test)
#    model.fit(x_train, y_train)
#    train_acc.append(model.score(x_train, y_train))
#    test_acc.append(model.score(x_test, y_test))
    forest.fit(x_train, y_train)
    train_acc.append(forest.score(x_train, y_train))
    test_acc.append(forest.score(x_test, y_test))
    
plt.plot(range(2,200,10),train_acc,label='training accuracy')
plt.plot(range(2,200,10),test_acc,label='test accuracy')
plt.ylabel('Accuracy')             
plt.xlabel('Principal Component Number')
plt.legend()


###LLE+RF/LR
train_acc=[]
test_acc=[]

for i in range(2,200,10):
    x_train,x_test,y_train,y_test=train_test_split(X1,y1,test_size=0.3,random_state=0)
    embedding = LocallyLinearEmbedding(n_components=i)
    x_train=embedding.fit_transform(x_train)
    x_test=embedding.transform(x_test)
    forest.fit(x_train, y_train)
    train_acc.append(forest.score(x_train, y_train))
    test_acc.append(forest.score(x_test, y_test))
#    model.fit(x_train, y_train)
#    train_acc.append(model.score(x_train, y_train))
#    test_acc.append(model.score(x_test, y_test))

plt.plot(range(2,200,10),train_acc,label='training accuracy')
plt.plot(range(2,200,10),test_acc,label='test accuracy')
plt.ylabel('Accuracy')             
plt.xlabel('Principal Component Number')    
plt.legend()


###ISOMAP+RF/LR
train_acc=[]
test_acc=[]

for i in range(2,200,10):
    x_train,x_test,y_train,y_test=train_test_split(X1,y1,test_size=0.3,random_state=0)
    embedding = Isomap(n_components=i)
    x_train=embedding.fit_transform(x_train)
    x_test=embedding.transform(x_test)
    forest.fit(x_train, y_train)
    train_acc.append(forest.score(x_train, y_train))
    test_acc.append(forest.score(x_test, y_test))
#    model.fit(x_train, y_train)
#    train_acc.append(model.score(x_train, y_train))
#    test_acc.append(model.score(x_test, y_test))

plt.plot(range(2,200,10),train_acc,label='training accuracy')
plt.plot(range(2,200,10),test_acc,label='test accuracy')
plt.ylabel('Accuracy')             
plt.xlabel('Principal Component Number')    
plt.legend()



