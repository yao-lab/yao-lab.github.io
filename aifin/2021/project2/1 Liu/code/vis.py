import torch
import dataset
from model import conv3
from torch.utils.data import Dataset, DataLoader
from utils import *
import pandas as pd
import  os 
import argparse
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser('Pretrain Network')
parser.add_argument('--weights_name', type=str, default='1e-4_1e-3_1000.pth',
        help='folder to store weights')
parser.add_argument('--batch_size', type=int, default=1,
        help='batch size')
parser.add_argument('--lr_initial', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--image_name', default='both.png', type=str, help='the tested image name')
parser.add_argument('--save_name', default='grad_cam.png', type=str, help='saved image name')
parser.add_argument('--model', type=str, default='resnet12')
args = parser.parse_args()

batch_size = args.batch_size
model_name = args.weights_name
device = 'cpu'
model = conv3()
file_list = os.listdir(os.path.join(os.getcwd(), 'monthly_20d'))
img_list = [i for i in file_list if 'dat' in i]
lab_list = [i for i in file_list if 'feather' in i]
img_list.sort()
lab_list.sort()
img_train_val, img_test = img_list[:7],img_list[7:] 
lab_train_val, lab_test = lab_list[:7],lab_list[7:] 
IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96} 
model.load_state_dict(torch.load(model_name))
model.eval()
i, j = img_test[0], lab_test[0]
img_test_arr = np.memmap(os.path.join(os.getcwd(), 'monthly_20d',i), dtype=np.uint8, mode='r').reshape(
                                (-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20]))
lab_test_arr = pd.read_feather(os.path.join(os.getcwd(), 'monthly_20d',j))['Ret_20d'].values
#testset = dataset.NpyDataset(img_test_arr, lab_test_arr)
#test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=4,
#                pin_memory=True, drop_last=True)
os.makedirs('pics/', exist_ok=True)



import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19

from gradcam import GradCam




IMAGE_NAME = args.image_name
SAVE_NAME = args.save_name


grad_cam = GradCam(model)

#for i in model.named_parameters():
#   print(i)
#model.eval()
model.load_state_dict(torch.load(model_name, map_location='cpu'))
for i in range(8):
    or_img =  Image.fromarray(img_test_arr[i])
    or_img.save(str(i) + IMAGE_NAME)
    test_image = (transforms.ToTensor()(Image.open(str(i) +IMAGE_NAME))).unsqueeze(dim=0)
    feature_image = grad_cam(test_image).squeeze(dim=0)
    feature_image = transforms.ToPILImage()(feature_image)
    feature_image.save(str(i) +SAVE_NAME)


    
    
