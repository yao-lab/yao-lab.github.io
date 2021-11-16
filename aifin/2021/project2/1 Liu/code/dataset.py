import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
class NumpyDataset(Dataset):
    def __init__(self, data_file_path, lab_file_path, binary=True):
        self.data = np.load(data_file_path)
        self.label = np.load(lab_file_path)
        self.binary = binary
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        x = torch.from_numpy(x).float()
        x = x.unsqueeze(0)
        if self.binary:
            y = np.where(y > 0, 1, 0)
       
        y = torch.from_numpy(y).float()
        return x, y
    
    def __len__(self):
        return len(self.data)
class NpyDataset(Dataset):
    def __init__(self, data_file, lab_file, binary=True):
        self.data = data_file
        self.label = lab_file
        self.binary = binary
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        x = torch.from_numpy(x).float()
        x = x.unsqueeze(0)
        if self.binary:
            y = np.where(y > 0, 1, 0)
       
        y = torch.from_numpy(y).float()
        return x, y
    
    def __len__(self):
        return len(self.data)
