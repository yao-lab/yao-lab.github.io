import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import real_data_target

def noise(quantity, size):
    return Variable(torch.randn(quantity, size))

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self, out_features, leakyRelu=0.2, hidden_layers=[128, 256], in_features=64, escalonate=False):
        super(GeneratorNet, self).__init__()
        
        self.in_features = in_features
        self.layers = hidden_layers.copy()
        self.layers.insert(0, self.in_features)

        for count in range(0, len(self.layers)-1):
            self.add_module("hidden_" + str(count), 
                nn.Sequential(
                    nn.Linear(self.layers[count], self.layers[count+1]),
                    nn.LeakyReLU(leakyRelu)
                )
            )

        if not escalonate:
            self.add_module("out", 
                nn.Sequential(
                    nn.Linear(self.layers[-1], out_features),
                    #torch.nn.Sigmoid()
                    nn.Tanh()
                    #nn.ReLU()
                )
            )
        else:
            self.add_module("out", 
                nn.Sequential(
                    nn.Linear(self.layers[-1], out_features),
                    #torch.nn.Sigmoid()
                    nn.Tanh()
                    #nn.ReLU()
                )
            )
    
    def forward(self, x):
        for name, module in self.named_children():
            x = module(x)
        return x

    def create_data(self, quantity):
        points = noise(quantity, self.in_features)
        #try:
            #data=self.forward(points.cuda())
        #except RuntimeError:
        data=self.forward(points.cpu())
        return data.detach().numpy()

def train_generator(optimizer, discriminator, loss, fake_data):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error