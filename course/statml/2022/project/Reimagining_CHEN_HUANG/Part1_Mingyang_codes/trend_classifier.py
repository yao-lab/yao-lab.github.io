from unittest import TestLoader
import pandas as pd
import numpy as np
import os.path as op
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from zmq import device
from models import SelfNet
from torchsummary import summary
import argparse
import os
import torch.optim as optim
import torch.nn as nn
from utils import *

IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}  

parser = argparse.ArgumentParser(description='trend classifier')
parser.add_argument('--start-year', type=int, default=1993, help='start year of training and val data')
parser.add_argument('--period', type=int, default=8, help='period of year for training and val data')
parser.add_argument('--gpu-num', type=int, default=0, help='gpu number')
parser.add_argument('--continue-train', type=str2bool, default=False)
parser.add_argument('--predict-days', type=int, default=20, help='number of days for prediction return trend')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--x-init', type=str2bool, default=True, help='xavier initial for weights')
parser.add_argument('--bn', type=str2bool, default=True, help='use batch norm')
parser.add_argument('--first-cnn', type=int, default=64, help='out channel of 1st cnn')
parser.add_argument('--layers', type=int, default=3, help='number of cnn layers')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio for mlp')
parser.add_argument('--activation', type=str, default='lrelu', choices=['relu','lrelu'], help='activation function')
parser.add_argument('--mp-size', type=str2tuple, default=(2,1), help='max pooling size')
parser.add_argument('--flt-size', type=str2tuple, default=(5,3), help='cnn kernel size')
parser.add_argument('--dilation', type=str2tuple, default=(2,1), help='cnn dilation size')
parser.add_argument('--stride', type=str2tuple, default=(1,1), help='cnn stride size')
parser.add_argument('--regular-store', type=str2bool, default=False)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%args.gpu_num
model_name = 'rgstore%s_start%s_period%s_preds%s_lr%s_epoch%s_bn%s_xinit%s_1stout%s_layers%s_drop%s_%s_mp%s_flt%s_dlt%s_strd%s' \
%(args.regular_store, args.start_year, args.period, args.predict_days, args.lr, args.epoch, args.bn, args.x_init,args.first_cnn,
args.layers, args.dropout, args.activation, args.mp_size, args.flt_size, args.dilation, args.stride)

# print('shishi')
# if True:
#     raise ValueError('shishi')

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

    # train_idx = list(np.random.choice(np.arange(0,793019),size=int(793019*0.7),replace=False))
    # test_idx = list(set(list(np.arange(0,793019))).difference(set(train_idx)))
    # f1= open("train_idx.txt","w")   
    # f1.write(str(train_idx)[1:-1])
    # f1.close() 
    # f2= open("test_idx.txt","w")   
    # f2.write(str(test_idx)[1:-1])  
    # f2.close() 

    # train_idx = list(np.arange(0,int(793019*0.7)))
    # test_idx = list(set(list(np.arange(0,793019))).difference(set(train_idx)))


    print('train_size:',len(train_idx))
    print('test_size',len(test_idx))
    
    train_x = data_x[train_idx]
    train_y = data_y[train_idx]
    test_x = data_x[test_idx]
    test_y = data_y[test_idx]

        
    train_dataset = TensorDataset(train_x,train_y)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    test_dataset = TensorDataset(test_x,test_y)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return train_loader, test_loader


def construct_model():
    if args.continue_train:
        state = torch.load('/home/mchen/hkust_coursework/MATH_5470/final_project/checkpoints/%s'%model_name)
        # print(state.keys())
        model = state['net']
        log = state['log']
    else:
        model = SelfNet(64,60,args.first_cnn,args.layers,args.dropout,args.bn,args.activation,args.mp_size
        ,args.flt_size,args.dilation,args.stride)
        if args.x_init:
            model._weights_init()
            # model.apply(weights_init)
        # The log for recording train (test) loss and errors.
        log = {
            'num_params': [],
            'train_loss': [],
            'train_error': [],
            'test_loss': [],
            'test_error': []
        }

        number_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log['num_params'].append(number_params)

        print(f'total parameters: {number_params}')

    use_gpu = torch.cuda.is_available()

    device = torch.device("cuda" if use_gpu else "cpu")
    print(device)

    # if use_gpu:
    model = model.to(device)
    
    return model,log,device

if __name__ == '__main__':
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    log_path = os.path.join('./logs',model_name)+'.log'
    sys.stdout = Logger(log_path)

    train_loader,test_loader = generate_loaders()
    model,log,devc = construct_model()

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loaders = {'train':train_loader,'test':test_loader}

    model, log = train_model(loaders, model, criterion, optimizer, log, model_name, args.regular_store, device=devc, num_epochs=args.epoch)