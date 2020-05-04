import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.nn.init import xavier_normal_ as xavier_normal_
from torch.nn.init import kaiming_normal_ as kaiming_normal_
from collections import OrderedDict

## Some candidate activations
class myact_Ramp(nn.Module):

    def __init__(self):
        super(myact_Ramp, self).__init__()

    def forward(self, x):
        return .5 * (F.hardtanh(input, min_val=-0.5, max_val=0.5) + 1)

class myact_Sin(nn.Module):
    def __init__(self):
        super(myact_Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)
    
class myact_LogReLU(nn.Module):
    def __init__(self):
        super(myact_LogReLU, self).__init__()

    def forward(self, x):
        return torch.log(1 + F.relu(x))

class myact_Sqrt(nn.Module):
    def __init__(self):
        super(myact_Sqrt, self).__init__()

    def forward(self, x):
        return torch.sqrt(1 + F.relu(x))
        


class GeneratorXi(nn.Module):
    def __init__(self, activation=None, hidden_units=None, input_dim=None):
        ## activation, 'Sigmoid'/'ReLU'/'LeakyReLU'
        ## transform, 'abs'/'exp'

        super(GeneratorXi, self).__init__()
        self.arg = {'negative_slope':0.2} if (activation == 'LeakyReLU') else {}
        self.activation = activation
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.layers = len(self.hidden_units)
        self.map = self._make_layers()


    def _make_layers(self):
        
        layer_list = []
        for lyr in range(self.layers):
            if lyr == 0:
                layer_list += [('lyr%d'%(lyr+1), nn.Linear(self.input_dim, self.hidden_units[lyr])),
                               ('act%d'%(lyr+1), getattr(nn, self.activation)(**self.arg))]
            else:
                layer_list += [('lyr%d'%(lyr+1), nn.Linear(self.hidden_units[lyr-1], self.hidden_units[lyr])),
                               ('act%d'%(lyr+1), getattr(nn, self.activation)(**self.arg))]

        layer_list += [('lyr%d'%(self.layers+1), nn.Linear(self.hidden_units[-1], 1))]

        return nn.Sequential(OrderedDict(layer_list))


    def forward(self, z):
        
        xi = self.map(z.view(-1, self.input_dim))
        xi = torch.abs(xi)    

        return xi  

## Generator W/Wb/b
class Generator(nn.Module):
    def __init__(self, p, elliptical=False, use_bias=False):
        ## activation, 'Sigmoid'/'ReLU'/'LeakyReLU'
        ## transform, 'abs'/'exp'

        super(Generator, self).__init__()
        self.p = p
        self.weight = nn.Parameter(torch.eye(self.p))
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.p))
        self.elliptical = elliptical

    def forward(self, z, xi=None):
        
        x = z.view(-1, self.p).mm(self.weight) ## cov W'W
        if self.elliptical:
            x = xi * x
        if self.use_bias:
            x = x + self.bias
        return x.view(-1, self.p)

class Discriminator(nn.Module):
    def __init__(self, p, hidden_units, activation, activation_1='Sigmoid',
                 activation_n='ReLU', prob=False):
        ## hidden_units: list of hidden units
        ## activation: Sigmoid/ReLU/LeakyReLU
        ## activation1: Ramp/Sigmoid/ReLU
        super(Discriminator, self).__init__()
        self.p = p
        self.prob = prob

        if activation not in ['ReLU', 'Sigmoid', 'LeakyReLU']:
            raise NameError('Activation is not defined!')
        self.arg = {'negative_slope':0.2} if (activation == 'LeakyReLU') else {}
        self.activation = activation

        if activation_1 not in ['Sqrt', 'Ramp', 'LogReLU', 'Sin', 'Sigmoid', 'ReLU', 'LeakyReLU']:
            raise NameError('Activation I is not defined!')
        self.arg_1 = {'negative_slope':0.2} if (activation_1 == 'LeakyReLU') else {}
        self.activation_1 = activation_1
        
        if activation_n not in ['Sqrt', 'Ramp', 'LogReLU', 'Sin', 'Sigmoid', 'ReLU', 'LeakyReLU']:
            raise NameError('Activation Last is not defined!')
        self.arg_n = {'negative_slope':0.2} if (activation_n == 'LeakyReLU') else {}
        self.activation_n = activation_n

        self.layers = len(hidden_units)
        self.hidden_units = hidden_units

        self.feature = self._make_layers()
        self.map_last = nn.Linear(self.hidden_units[-1], 1)
        if self.prob:
            self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.feature(x.view(-1,self.p)) # allow 1-dim [N,]
        d = self.map_last(x).squeeze() #[N,]
        if self.prob:
            d = self.sig(d)
        return x, d

    def _make_layers(self):
        layer_list = []
        for lyr in range(self.layers):
            if lyr == 0:
                if self.activation_1 in ['Sigmoid', 'ReLU', 'LeakyReLU']:
                    layer_list += [('lyr%d'%(lyr+1), nn.Linear(self.p, self.hidden_units[lyr])),
                                   ('act%d'%(lyr+1), getattr(nn, self.activation_1)(**self.arg_1))]
                
                elif self.activation_1 in ['Sqrt', 'Ramp', 'Sin', 'LogReLU']:
                    layer_list += [('lyr%d'%(lyr+1), nn.Linear(self.p, self.hidden_units[lyr])),
                                   ('act%d'%(lyr+1), eval('myact_'+self.activation_1)())]

            elif lyr == (self.layers - 1):
                if self.activation_n in ['Sigmoid', 'ReLU', 'LeakyReLU']:
                    layer_list += [('lyr%d'%(lyr+1), nn.Linear(self.hidden_units[lyr-1], self.hidden_units[lyr])),
                                   ('act%d'%(lyr+1), getattr(nn, self.activation_n)(**self.arg_n))]
                elif self.activation_n in ['Sqrt', 'Ramp', 'Sin', 'LogReLU']:
                    layer_list += [('lyr%d'%(lyr+1), nn.Linear(self.hidden_units[lyr-1], self.hidden_units[lyr])),
                                   ('act%d'%(lyr+1), eval('myact_'+self.activation_n)())]

            else:
                layer_list += [('lyr%d'%(lyr+1), nn.Linear(self.hidden_units[lyr-1], self.hidden_units[lyr])),
                               ('act%d'%(lyr+1), getattr(nn, self.activation)(**self.arg))]

        return nn.Sequential(OrderedDict(layer_list))


class PoolSet(Dataset):
    
    def __init__(self, p_x):
        ## input: torch.tensor (NOT CUDA TENSOR)
        self.len = len(p_x)
        self.x = p_x # [N, p]
    
    def __getitem__(self, index):
        return self.x[index]
    
    def __len__(self):
        return self.len

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
      m.weight.data.normal_(0.0, 0.02)
      m.bias.data.fill_(0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
      xavier_normal_(m.weight)
      m.bias.data.fill_(0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
      kaiming_normal_(m.weight)
      m.bias.data.fill_(0.0)

