# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:57:40 2022

@author: MXR
"""
import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%
year_list = np.arange(1993,2001,1) #1993-2000 as training & validation set

images = []
label_df = []
for year in tqdm(year_list):
    images.append(np.memmap(os.path.join("E:/Study/HKUST/2-Statistical Prediction/Final_Project_Image/monthly_20d",\
                                         f"20d_month_has_vb_[20]_ma_{year}_images.dat"), dtype=np.uint8, mode='r')
                   .reshape((-1, 64, 60)))
    label_df.append(pd.read_feather(os.path.join("E:/Study/HKUST/2-Statistical Prediction/Final_Project_Image/monthly_20d",\
                                                 f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather")))
    
images = np.concatenate(images)
label_df = pd.concat(label_df)

print(images.shape)
print(label_df.shape)

#%%
img = images[120,:,:]
fd, hog_image = hog(img, 
                    orientations=9, 
                    pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), 
                    block_norm = 'L1',
                    visualize=True, 
                    multichannel=False)
plt.figure()
plt.axis("off")
plt.imshow(hog_image, cmap="gray")

plt.figure()
plt.axis("off")
plt.imshow(img, cmap="gray")

#%%
feature_hog = []
for i in tqdm(np.arange(np.shape(images)[0])):
    feature_hog.append(hog(images[i,:,:], 
                            orientations=9, 
                            pixels_per_cell=(8, 8),
                        	cells_per_block=(2, 2), 
                            block_norm = 'L1',
                            visualize=False, 
                            multichannel=False))
#%%
# feature_hog = np.array(feature_hog)

# np.save('E:/Study/HKUST/2-Statistical Prediction/Final_Project_Image/HOG_feature.npy',feature_hog)

#%%
feature_hog = np.load('E:/Study/HKUST/2-Statistical Prediction/Final_Project_Image/HOG_feature.npy')
#%%
y = label_df['Retx_20d_label'].to_numpy()
tmp = (np.isnan(y) | (y==2))
y = y[tmp==False]

feature_hog = feature_hog[tmp==False, :]
#%%
train_size = np.int(np.shape(y)[0]*5/7)

X_train = feature_hog[:train_size, :]
y_train = y[:train_size]


X_val = feature_hog[train_size:, :]
y_val = y[train_size:]

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

# clf_LR = LogisticRegression(penalty = 'l2', random_state = 0, C = 10**3,\
#                             verbose = 2)
# clf_LR.fit(X_train, y_train)


# clf_Ada = AdaBoostClassifier(random_state = 0, 
#                              n_estimators = 10, 
#                              learning_rate=0.05)
# clf_Ada.fit(X_train, y_train)

# clf_RF = RandomForestClassifier(n_estimators=100, 
#                                 criterion = "gini", 
#                                 # n_jobs = 4, 
#                                 random_state = 0)
# clf_RF.fit(X_train, y_train)
#%%
def get_metrics(labels, predict):
    print("Accuracy: ", accuracy_score(labels, predict))
    print("Precision: ", precision_score(labels, predict))
    print("Recall: ", recall_score(labels, predict))
    print("F1-score: ", f1_score(labels, predict))
    
def plot_roc(labels, predict_prob):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    plt.figure()
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()
    
### Logistic Regression  
y_proba = clf_LR.predict_proba(X_val)[:,1]
y_pred = clf_LR.predict(X_val)

### Adaboost
y_proba = clf_Ada.predict_proba(X_val)[:,1]
y_pred = clf_Ada.predict(X_val)

### Random Forest
y_proba = clf_RF.predict_proba(X_val)[:,1]
y_pred = clf_RF.predict(X_val)

### Random Guess
y_proba = np.random.normal(loc=0.5, scale=0.1, size=np.shape(y_val))
y_pred = (y_proba >= 0.5)

plot_roc(y_val, y_proba)
get_metrics(y_val, y_pred)


# print(np.sum(0==y_val)/np.shape(y_val)[0])

# features = ft.hog(image,  # input image
#                   orientations=ori,  # number of bins
#                   pixels_per_cell=ppc, # pixel per cell
#                   cells_per_block=cpb, # cells per blcok
#                   block_norm = 'L1', #  block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}, optional
#                   transform_sqrt = True, # power law compression (also known as gamma correction)
#                   feature_vector=True, # flatten the final vectors
#                   visualise=False)
#%%



