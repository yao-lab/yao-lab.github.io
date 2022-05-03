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

from model_debug import MAE_ViT_Classifier, MAE_ViT


IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}  

parser = argparse.ArgumentParser(description='trend classifier')
parser.add_argument('--start-year', type=int, default=1993, help='start year of training and val data')
parser.add_argument('--period', type=int, default=8, help='period of year for training and val data')
parser.add_argument('--gpu_num', type=int, default=1, help='gpu number')
parser.add_argument('--predict_days', type=int, default=5, help='number of days for prediction return trend')

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=512)

parser.add_argument('--num_classes', type=int, default=2)

parser.add_argument('--base_learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--mask_ratio', type=float, default=0.75)
parser.add_argument('--total_epoch', type=int, default=50)
parser.add_argument('--model_path', type=str, default='m20d-vit-t-MAE-classifier.pt')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%args.gpu_num

# python finetuning_classifier.py --mask_ratio 0.25 --predict_days 5 --gpu_num 0

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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mae_model = MAE_ViT(mask_ratio=args.mask_ratio).cuda()
    mae_model_path = 'mask_ratio(%s)-width(6)-m20d-vit-t-mae.pt'%str(args.mask_ratio)

    mae_model.load_state_dict(torch.load(mae_model_path).state_dict())

    model = MAE_ViT_Classifier(mae_model, args.num_classes).to(device)

    optim = torch.optim.Adam(model.fc.parameters(), lr=args.base_learning_rate)
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())
    
    root_path = 'MAE_fintuning-mask_ratio(%s)'%str(args.mask_ratio) + '-width(6)' + '-' + 'predcited_days(%s)'%args.predict_days
    if not os.path.isdir(root_path):
        os.makedirs(root_path)
    model_path = os.path.join(root_path, args.model_path)
    log_csv_path = os.path.join(root_path, 'results_log.csv')

    step_count = 0
    best_val_acc = 0
    optim.zero_grad()


    for e in range(args.total_epoch):
        model.train()
        losses = []
        acces = []
        for img, label in tqdm(iter(train_loader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = nn.CrossEntropyLoss()(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()

        losses.append(loss.item())
        acces.append(acc.item())

        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for img, label in tqdm(iter(test_loader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = nn.CrossEntropyLoss()(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  
       

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'saving best model with acc {best_val_acc} at {e} epoch!')       
            torch.save(model, model_path)

        with open(log_csv_path, 'a') as f:
            f.write('%3d, %.5f, %.5f, %.5f, %.5f \n' % (e, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc))

    with open(log_csv_path, 'a') as f:
            f.write('best val acc:, %.5f  \n' % (best_val_acc))



