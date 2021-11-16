import torch.nn as nn
import torch

class OURCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(OURCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(5,3),stride=(3,1),dilation=(2,1),padding=(67,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(5,3),stride=(3,1),dilation=(2,1),padding=(35,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(5,3),stride=(3,1),dilation=(2,1),padding=(19,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1))
            )

        for x in self.features:
            if isinstance(x, nn.Conv2d):
                torch.nn.init.xavier_uniform_(x.weight)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256*8*60, out_features=2),
            nn.Dropout(0.5)
        )

        for x in self.classifier:
            if isinstance(x, nn.Linear):
                torch.nn.init.xavier_uniform_(x.weight)

   
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        # self.classifier.apply(self.init_weights)
        return x