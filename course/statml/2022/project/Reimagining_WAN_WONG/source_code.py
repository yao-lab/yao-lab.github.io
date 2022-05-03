import torch
import pandas as pd
import numpy as np
import os.path as op
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from torchvision import transforms

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics

from pytorch_lightning.callbacks.early_stopping import EarlyStopping


IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}        

def train_val_dataset(dataset, val_split=0.3):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


#  Here we just collect the 20-day version.
class PriceDataset(Dataset):
    """Face Landmarks dataset."""
    
    def get_image_by_year(self, year):
        return np.memmap(op.join("./monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_images.dat"), dtype=np.uint8, mode='r').reshape(
                            (-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20]))

    def get_feather_by_year(self, year):
        return pd.read_feather(op.join("./monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"))

    def __init__(self, year):
        self.image = self.get_image_by_year(year)
        self.label_df = self.get_feather_by_year(year)
            
    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):

        sample = {'image': self.image[idx], 'label': 1 if self.label_df.iloc[idx]["Ret_20d"] > 0 else 0 , 'ret': self.label_df.iloc[idx]["Ret_20d"] }
        return sample

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.conv1 = nn.Conv2d(3, 64 , (5, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 1), stride=(1, 3), dilation=(2, 3))
        self.conv2 = nn.Conv2d(64, 128 , (5, 3))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 1))
        self.conv3 = nn.Conv2d(128, 256 , (5, 3))
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((2, 1))

        self.fc1 = nn.Linear(45056,1)
        self.dropout = nn.Dropout(p=0.5)

        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.xavier_uniform(self.conv2.weight)
        nn.init.xavier_uniform(self.conv3.weight)

    def forward(self, x):
        #print("-----> Shape:",x.shape)
        x = nn.LeakyReLU()(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = nn.LeakyReLU()(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = nn.LeakyReLU()(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.dropout(x)

        output = nn.Sigmoid()(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        x = x / 255

        # Convert to 3D image
        x = x.repeat(3, 1, 1)
        x = x.reshape(-1,3,64,60)   
        y_hat = self(x)
        y_hat = y_hat.reshape(-1).float()
        loss = F.binary_cross_entropy(y_hat.float(), y.float())
        self.train_acc((y_hat>0.5).int(), y.int())
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.train_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        x = x / 255

        # Convert to 3D image
        x = x.repeat(3, 1, 1)
        x = x.reshape(-1,3,64,60)   
        y_hat = self(x)
        y_hat = y_hat.reshape(-1).float()
        loss = F.binary_cross_entropy(y_hat.float(), y.float())
    
        self.val_acc((y_hat>0.5).int(), y.int())
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log('val_acc_epoch', self.val_acc, prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        x = x / 255

        # Convert to 3D image
        x = x.repeat(3, 1, 1)
        x = x.reshape(-1,3,64,60)   
        y_hat = self(x)
        y_hat = y_hat.reshape(-1).float()
        loss = F.binary_cross_entropy(y_hat.float(), y.float())
        self.test_acc((y_hat>0.5).int(), y.int())
        self.log("test_loss", loss)
        return loss

    def test_epoch_end(self, outs):
        self.log('test_acc_epoch', self.test_acc)



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

if __name__ == "__main__":
    

    all_training_data = ConcatDataset([PriceDataset(i) for i in range(1993,2000)])
    training_data = train_val_dataset(all_training_data)

    train_data = training_data['train']
    val_data = training_data['val']
    test_data = ConcatDataset([PriceDataset(i) for i in range(2000,2018)])
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=128)
    test_dataloader = DataLoader(test_data, batch_size=128)
    
    logger = TensorBoardLogger("tb_logs", name="my_model")

    trainer = pl.Trainer(devices=[6,7,8,9],accelerator='gpu', logger=logger, callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
    model = LitModel()
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)