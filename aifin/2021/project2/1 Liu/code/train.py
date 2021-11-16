import torch
import dataset
from model import conv3
from torch.utils.data import Dataset, DataLoader
from utils import *
import pandas as pd
import  os 
import argparse
parser = argparse.ArgumentParser('Pretrain Network')
parser.add_argument('--weights_name', type=str, default='weight.pth',
        help='folder to store weights')
parser.add_argument('--batch_size', type=int, default=128,
        help='batch size')
parser.add_argument('--lr_initial', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
args = parser.parse_args()

train_x , train_y = np.load('monthly_20d/train_x.npy'), np.load('monthly_20d/train_y.npy')
val_x , val_y = np.load('monthly_20d/val_x.npy'), np.load('monthly_20d/val_y.npy')
trainset = dataset.NpyDataset(train_x , train_y )
valset = dataset.NpyDataset(val_x , val_y)
batch_size = args.batch_size
epoch = 15
lr = args.lr_initial
wd =  args.weight_decay
model_name = args.weights_name
train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=4,
                pin_memory=True, drop_last=True)
val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, num_workers=4,
                pin_memory=True, drop_last=True)

device = 'cuda'
model = conv3().cuda()
optimizer=torch.optim.AdamW(model.parameters(), lr, weight_decay=wd)
evaluate_batch(model, val_loader, device)
best_acc = 0
count = 0

file_list = os.listdir(os.path.join(os.getcwd(), 'monthly_20d'))
img_list = [i for i in file_list if 'dat' in i]
lab_list = [i for i in file_list if 'feather' in i]
img_list.sort()
lab_list.sort()
img_train_val, img_test = img_list[:7],img_list[7:] 
lab_train_val, lab_test = lab_list[:7],lab_list[7:] 
IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96} 

for i in range(1, epoch+1):
    print('Epoch : ', i)
    pretrain(model, train_loader, optimizer, device)
    val_acc = evaluate_batch(model, val_loader, device)
    print('Val Acc : ', val_acc)
    if  best_acc < val_acc:
        count = 0
        best_acc = val_acc
        torch.save(model.state_dict(), model_name)
    else:
        count += 1
    if count >= 2:
        break

model.load_state_dict(torch.load(model_name))
model.eval()
N1, N2 =0, 0
for i, j in zip(img_test, lab_test):
    img_test_arr = np.memmap(os.path.join(os.getcwd(), 'monthly_20d',i), dtype=np.uint8, mode='r').reshape(
                                (-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20]))
    lab_test_arr = pd.read_feather(os.path.join(os.getcwd(), 'monthly_20d',j))['Ret_20d'].values
    testset = dataset.NpyDataset(img_test_arr, lab_test_arr)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=4,
                pin_memory=True, drop_last=True)
    test_acc = evaluate_batch(model, test_loader, device)
    N2 += len(testset)
    N1 += (test_acc *len(testset))
print('Test Acc : ', test_acc)
