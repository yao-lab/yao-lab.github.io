import numpy as np
import  os 
import pandas as pd
file_list = os.listdir(os.getcwd())
img_list = [i for i in file_list if 'dat' in i]
lab_list = [i for i in file_list if 'feather' in i]
img_list.sort()
lab_list.sort()
print(img_list)
print(lab_list)
img_train_val, img_test = img_list[:7],img_list[7:] 
lab_train_val, lab_test = lab_list[:7],lab_list[7:] 
IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96} 
img_train_val_arr, img_test_arr = [], []
lab_train_val_arr, lab_test_arr = [], [] 
for i, j in zip(img_train_val, lab_train_val):
    img_train_val_arr.append(np.memmap(i, dtype=np.uint8, mode='r').reshape(
                                (-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20])))
    lab_train_val_arr.append(pd.read_feather(j)['Ret_20d'].values)
for i, j in zip(img_test, lab_test):
    img_test_arr.append(np.memmap(i, dtype=np.uint8, mode='r').reshape(
                                (-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20])))
    lab_test_arr.append(pd.read_feather(j)['Ret_20d'].values)
img_train_val_data = np.concatenate(img_train_val_arr, 0)/255.0
lab_train_val_data = np.concatenate(lab_train_val_arr, 0)
img_train_data, img_val_data = img_train_val_data[:int(0.7*len(img_train_val_data))], img_train_val_data[int(0.7*len(img_train_val_data)):]
lab_train_data, lab_val_data = lab_train_val_data[:int(0.7*len(lab_train_val_data))], lab_train_val_data[int(0.7*len(lab_train_val_data)):]
img_test_data = np.concatenate(img_test_arr, 0)/255.0
lab_test_data = np.concatenate(lab_test_arr, 0)
print(np.unique(img_train_val_data))
print(img_train_data.shape)
print(lab_train_data.shape)
print(img_val_data.shape)
print(lab_val_data.shape)
print(img_test_data.shape)
print(lab_test_data.shape)
np.save('train_x.npy', img_train_data)
np.save('train_y.npy', lab_train_data)
np.save('val_x.npy', img_val_data)
np.save('val_y.npy', lab_val_data)
np.save('test_x.npy', img_test_data)
np.save('test_y.npy', lab_test_data)



