import pandas as pd
import numpy as np
import os.path as op
import torch
import torch.utils.data as Data
from torch.utils.data import TensorDataset


class EqDataset(TensorDataset):
    def __init__(self, data_dir, sd,ed,transform=None):
        self.data_info = self.get_img_info(data_dir,sd,ed)
        self.transform = transform

    def __getitem__(self, index):
        img = self.data_info[0][index]
        label = self.data_info[1][index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.data_info[0].shape[0]

    @staticmethod
    def get_img_info(data_dir,sd,ed):
        data_info = list()
        IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
        IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}
        directory_folder = op.join(data_dir,'monthly_20d')
        X_images = np.concatenate([np.memmap(op.join(directory_folder, f"20d_month_has_vb_[20]_ma_{i}_images.dat"), dtype=np.uint8, mode='r').reshape(
                        (-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20]))for i in range(sd,ed)],axis=0)
        Y_label = pd.concat([pd.read_feather(op.join(directory_folder, f"20d_month_has_vb_[20]_ma_{i}_labels_w_delay.feather"))
                        for i in range(sd,ed)],axis=0)
        Y_label = (Y_label.iloc[:,5]>0).apply(lambda x:int(x)).to_numpy().reshape(-1,1)
        if( X_images.shape[0]!=Y_label.shape[0]):
            print("ERROR:The dimension of X and Y are not same!")
            exit()
        data_info=[X_images,Y_label]
        return data_info

