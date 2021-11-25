import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import real_data_target, fake_data_target

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self, in_features, leakyRelu=0.2, dropout=0.3, hidden_layers=[256, 128]):
        super(DiscriminatorNet, self).__init__()
        
        out_features = 1
        self.layers = hidden_layers.copy()
        self.layers.insert(0, in_features)

        for count in range(0, len(self.layers)-1):
            self.add_module("hidden_" + str(count), 
                nn.Sequential(
                    nn.Linear(self.layers[count], self.layers[count+1]),
                    nn.LeakyReLU(leakyRelu),
                    nn.Dropout(dropout)
                )
            )
        
        self.add_module("out", 
            nn.Sequential(
                nn.Linear(self.layers[-1], out_features),
                torch.nn.Sigmoid()
            )
        )

    def forward(self, x):
        for name, module in self.named_children():
            x = module(x)
        return x

# train_discriminator(d_optimizer, discriminator, loss, real_data, fake_data)
def train_discriminator(optimizer, discriminator, loss, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error
    return error_real + error_fake, prediction_real, prediction_fake