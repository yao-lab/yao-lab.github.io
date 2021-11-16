import torch
from torch import nn
import torch.nn.functional as F
class ConvNet(nn.Module):
    """Encoder for feature embedding"""
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size=(5,3), padding=(5, 3), stride=(3,1) ,dilation=(2,1)),
                        nn.LeakyReLU(0.1),
                        nn.MaxPool2d(2,1))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,kernel_size=(5,3),padding=(5,3), stride=(3,1), dilation=(2,1)),
                        nn.LeakyReLU(0.1),
                        nn.MaxPool2d((2,1)),)
        self.layer3 = nn.Sequential(
                        nn.Conv2d(128,256,kernel_size=(5,3),padding=(5,3), stride=(3,1), dilation=(2,1)),
                        nn.LeakyReLU(0.1),
                        nn.MaxPool2d((2,1)),)
        self.fc = nn.Linear(18176, 2)
        self.dropout = nn.Dropout(p=0.5)




    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        result = self.fc(out)
        return result

def conv3():
    return ConvNet()

