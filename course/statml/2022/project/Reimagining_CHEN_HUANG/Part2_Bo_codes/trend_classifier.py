from unittest import TestLoader
import pandas as pd
import numpy as np
import os.path as op
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import argparse
import os
import torch.optim as optim
import torch.nn as nn
from utils import *
import random

import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model_debug import *


IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}  

parser = argparse.ArgumentParser(description='trend classifier')
parser.add_argument('--start-year', type=int, default=1993, help='start year of training and val data')
parser.add_argument('--period', type=int, default=8, help='period of year for training and val data')
parser.add_argument('--gpu-num', type=int, default=3, help='gpu number')
# parser.add_argument('--continue-train', type=bool, default=False)
parser.add_argument('--predict-days', type=int, default=5, help='number of days for prediction return trend')
# parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# parser.add_argument('--epoch', type=int, default=300, help='epoch number')
# parser.add_argument('--x-init', type=bool, default=True, help='xavier initial for weights')
# parser.add_argument('--bn', type=bool, default=True, help='use batch norm')
# parser.add_argument('--first-cnn', type=int, default=64, help='out channel of 1st cnn')
# parser.add_argument('--layers', type=int, default=3, help='number of cnn layers')
# parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio for mlp')
# parser.add_argument('--activation', type=str, default='lrelu', choices=['relu','lrelu'], help='activation function')
# parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio for mlp')


parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--max_device_batch_size', type=int, default=1024)
parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--mask_ratio', type=float, default=0.75)
parser.add_argument('--total_epoch', type=int, default=1000)
parser.add_argument('--warmup_epoch', type=int, default=100)
parser.add_argument('--model_path', type=str, default='m20d-vit-t-mae.pt')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%args.gpu_num


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_loaders():
    year=args.start_year
    images = np.memmap(op.join("./monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_images.dat"), dtype=np.uint8, mode='r').reshape(
                            (-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20]))
    print(images.shape)
    years = np.arange(year+1,year+int(args.period))
    print(years)
    for y in years:
        img = np.memmap(op.join("./monthly_20d", f"20d_month_has_vb_[20]_ma_{y}_images.dat"), dtype=np.uint8, mode='r').reshape(
                            (-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20]))
        print('year:',y,'size:',img.shape)
        images = np.concatenate((images,img), axis=0)
    
    label_df = pd.read_feather(op.join("./monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"))
    rets = np.array(label_df['Ret_%sd'%args.predict_days])
    for y in years:
        lb_df = pd.read_feather(op.join("./monthly_20d", f"20d_month_has_vb_[20]_ma_{y}_labels_w_delay.feather"))
        ret = np.array(lb_df['Ret_%sd'%args.predict_days])
        print('year:',y,'size:',ret.shape)
        rets = np.append(rets,ret)
    print('final rets shape:',rets.shape)

    images = images/255.0
    rets[rets>0] = 1
    rets[rets!=1] = 0
    print('postive num:',sum(rets==1),'negative num:',sum(rets==0))

    data_x = torch.from_numpy(images).type(torch.FloatTensor)
    data_y = torch.from_numpy(rets).to(torch.int64) 

    data_x = torch.unsqueeze(data_x,dim=1)
    # data_y = torch.unsqueeze(data_y,dim=1)

    f1= open("train_idx.txt","r")   
    train_idx = f1.read()    
    f1.close() 
    f2= open("test_idx.txt","r")   
    test_idx = f2.read()    
    f2.close() 

    train_idx = train_idx.split(',')
    test_idx = test_idx.split(',')
    train_idx = [int(e) for e in train_idx]
    test_idx = [int(e) for e in test_idx]

    print('train_size:',len(train_idx))
    print('test_size',len(test_idx))
    
    train_x = data_x[train_idx]
    train_y = data_y[train_idx]
    test_x = data_x[test_idx]
    test_y = data_y[test_idx]

        
    train_dataset = TensorDataset(train_x,train_y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = TensorDataset(test_x,test_y)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


if __name__ == '__main__':
    train_dataset, test_dataset, train_loader,test_loader = generate_loaders()

    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    model_path = 'mask_ratio(%s)'%str(args.mask_ratio) + '-width(6)' + '-' + args.model_path 

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for img, label in tqdm(iter(train_loader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            # if step_count % 1 == 0:
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([test_dataset[i][0] for i in range(16)])
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        
        ''' save model '''

        if (e+1) % 50 == 0:
            torch.save(model, model_path)

