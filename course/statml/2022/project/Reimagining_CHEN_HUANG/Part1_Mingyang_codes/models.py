import torch.nn as nn
import torch.nn.functional as F

out_channels = [128,256,512]

class BasicBlock1(nn.Module):
    def __init__(self,in_channel:int,out_channel:int,bn:bool,activation:str,
                mp_size:tuple,kernel_size:tuple,dilation:tuple,stride:tuple):
        super(BasicBlock1,self).__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channel)
        if activation=='relu':
            self.relu1 = nn.ReLU(inplace=True)
        elif activation=='lrelu':
            self.relu1 = nn.LeakyReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(mp_size)

    def forward(self, x):
        if self.bn == True:
            y = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        else:
            y = self.pool1(self.relu1(self.conv1(x)))
        return y

class SelfNet(nn.Module):
    def __init__(self,data_h,data_w,first_cnn:int,layers:int,dropout,bn:bool,activation:str,
                mp_size:tuple,kernel_size:tuple,dilation:tuple,stride:tuple,block=BasicBlock1):
        super(SelfNet,self).__init__()
        if layers>4 or layers<2:
            raise ValueError('CNN layers should be >=2 & <=4!')
        self.layers = layers
        self.layers_list = []
        in_out_channels = [1,first_cnn]+out_channels
        for l in range(layers):
            # setattr(self,'cnn'+str(l+1), block(in_channel=in_out_channels[l], out_channel=in_out_channels[l+1], bn=bn, activation=activation,
            # mp_size=mp_size, kernel_size=kernel_size, dilation=dilation, stride=stride))
            self.layers_list.append(block(in_channel=in_out_channels[l], out_channel=in_out_channels[l+1], bn=bn, activation=activation,
            mp_size=mp_size, kernel_size=kernel_size, dilation=dilation, stride=stride))

        self.cnn = nn.Sequential(*self.layers_list)
        output_size = self._compute_output_size(data_h,data_w,layers,mp_size,kernel_size,dilation,stride)
        # self.fc1 = nn.Sequential(nn.Linear(output_size,2),nn.Dropout(dropout))
        
        self.fc1 = nn.Linear(output_size,2)
        # self.fc1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(256*54,2))
        # self.fc2 = nn.Sequential(nn.Dropout(0.5), nn.Linear(256,2))
    

    def _compute_output_size(self,data_h,data_w,layers,mp_size,kernel_size,dilation,stride):
        for _ in range(layers):
            data_h = int(((data_h-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1)
            data_h = int(((data_h-1*(mp_size[0]-1)-1)/mp_size[0])+1)
            data_w = int(((data_w-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1)
            data_w = int(((data_w-1*(mp_size[1]-1)-1)/mp_size[1])+1)
        return out_channels[layers-2]*data_h*data_w

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self,x):
        # for l in range(self.layers):
        #     y = getattr(self,'cnn'+str(l+1))(y)
        y = self.cnn(x)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        # y = self.fc2(y)
    
        return y
