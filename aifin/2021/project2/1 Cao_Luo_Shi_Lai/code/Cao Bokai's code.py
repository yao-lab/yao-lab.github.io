#!/usr/bin/env python
# coding: utf-8

##Overall train and val model
import pandas as pd
import numpy as np
import os.path as op

IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}
date_spread = 20
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class CNN_20D(nn.Module):


    def __init__(self):
        super(CNN_20D, self).__init__()
        self.cnn_block1 = CNN_Block(in_channels=1, out_channels=64, kernel_size=[5, 3], stride=[3, 1], padding=[7, 1]
                                    , pooling_size=[2, 1], dilation=[2, 1], negative_slope=0.01)
        self.cnn_block2 = CNN_Block(in_channels=64, out_channels=128, kernel_size=[5, 3]
                                    , pooling_size=[2, 1], negative_slope=0.01)
        self.cnn_block3 = CNN_Block(in_channels=128, out_channels=256, kernel_size=[5, 3]
                                    , pooling_size=[2, 1], negative_slope=0.01)
        self.linear = nn.Linear(
            in_features=46080, out_features=2)

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        #x = F.softmax(x,dim=1)
        return x


class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[5, 3], stride=1, padding=[2, 1], pooling_size=[2, 1],
                 dilation=1, negative_slope=0.01):
        super(CNN_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.max_pooling = nn.MaxPool2d(kernel_size=pooling_size)
        self.LReLU = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.LReLU(x)
        x = self.max_pooling(x)
        return x


def evaluate(model, loss_function, x, label):
    '''
    '''
    x = x[:500]
    label = label[:500]
    print('1')
    x = torch.Tensor(x).cuda().reshape(x.shape[0], 1, 64, 60)
    label = torch.tensor(label, dtype=torch.long).cuda().reshape(label.shape[0], 1)
    print('2')
    loss = 0
    total = 0
    correct = 0


    x_item = x[:, 0:1, :, :]
    label_item = label[:, 0]
    output = model(x_item)
    predict = output.argmax(dim=1)
    loss_item = loss_function(output, label_item).item()
    loss = loss + loss_item
    total = total + x.shape[0]
    correct = correct + (predict.cpu().numpy() == label_item.cpu().numpy()).sum()
    loss = loss / x.shape[1]
    acc = correct / total * 100
    return loss, acc


def train(model, images, label, images_93_00_validation, label_93_00_validation):
    min_eva_loss = 10000000
    epoch = 100
    batch_size = 64
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    input = np.array(images)
    num_batch = math.floor(input.shape[0] / batch_size)
    data = torch.Tensor(input).cuda()[:num_batch*batch_size].reshape(batch_size, num_batch, 64, 60)
    label = torch.tensor(label, dtype=torch.long).cuda()[:num_batch*batch_size].reshape(batch_size, num_batch)
    for i in range(epoch * num_batch):
        x = data[:, (i % num_batch):(i % num_batch) + 1, :, :]
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        label_item = label[:, (i % num_batch)]
        output = model(x)
        predict = output.argmax(dim=1)
        loss = loss_function(output, label_item)
        correct = (predict.cpu().numpy() == label_item.cpu().numpy()).sum()
        acc = correct / batch_size * 100
        loss.backward()
        optimizer.step()
        print('epoch:', math.floor(i/num_batch), '  {}/{}'.format(i%num_batch, num_batch), ' train loss:', loss.item(), ' accuracy rate:', acc)
        del x
        if i % num_batch == 0:
            eva_loss, eva_acc = evaluate(model, loss_function, images_93_00_validation, label_93_00_validation)
            print("Epoch Step: %d | Train Loss: %f | Accuracy rate %f%% | Test Loss: %f | Test Accuracy rate %f%%" %
                  (i, loss.item(), acc, eva_loss, eva_acc))
            if eva_loss < min_eva_loss:
                min_eva_loss = eva_loss
                torch.save(model, '/home/mafs6010g1/model_v2.pkl')
                print("save model!")

if __name__ == '__main__':
    years = list(range(1993, 2000))
    images_93_00 = np.array([])
    label_93_00 = np.array([])
    for year in years:
        images = np.memmap(op.join("/home/mafs6010g1/img_data/monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_images.dat"), dtype=np.uint8,
                           mode='r').reshape((-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20]))
        images = images/255
        label_df = pd.read_feather(op.join("/home/mafs6010g1/img_data/monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"))
        # label_df.head()
        label = label_df['Retx_{}d_label'.format(date_spread)].values
        label = label >= 1
        label = label + 0
        images = images[label != 2]
        label = label[label != 2]
        if year == 1993:
            images_93_00 = images
            label_93_00 = label
        else:
            images_93_00 = np.concatenate((images_93_00, images), axis=0)
            label_93_00 = np.concatenate((label_93_00, label), axis=0)
    model = CNN_20D().cuda()
    images_93_00, images_93_00_validation, label_93_00, label_93_00_validation = train_test_split(images_93_00, label_93_00, test_size=0.3)
    images_93_00_validation = images_93_00_validation[-500:]
    label_93_00_validation = label_93_00_validation[-500:]
    train(model, images_93_00, label_93_00, images_93_00_validation, label_93_00_validation)
