import torch
from torch import nn
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        # Fully Connected Layer 1
        self.layer1 = nn.Linear(18, 32)
        # Fully Connected Layer 2
        self.layer2 = nn.Linear(32, 16)
        # Fully Connected Layer 3
        self.layer3 = nn.Linear(16, 8)
        # Fully Connected Layer 4
        self.layer4 = nn.Linear(8, 4)
        # Fully Connected Layer 5
        self.layer5 = nn.Linear(4, 2)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers) # utilize the LSTM model in torch.nn 
        # self.forwardCalculation = nn.Linear(hidden_size, output_size)
 
    def forward(self, input_data):
        z1_ = self.layer1(input_data)
        z1 = torch.sigmoid(z1_)
        
        z2_ = self.layer2(z1)
        z2 = torch.sigmoid(z2_)

        z3_ = self.layer3(z2)
        z3 = torch.sigmoid(z3_)

        z4_ = self.layer4(z3)
        z4 = torch.sigmoid(z4_)

        z5_ = self.layer5(z4)
        return z5_