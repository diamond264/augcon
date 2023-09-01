# refer to :https://github.com/FengHZ/KD3A/blob/master/model/digit5.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10, mlp=True):
        super(CNN, self).__init__()
        encoder = nn.Sequential()
        encoder.add_module("conv1", nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2))
        encoder.add_module("bn1", nn.BatchNorm2d(64))
        encoder.add_module("relu1", nn.ReLU())
        encoder.add_module("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        encoder.add_module("conv2", nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        encoder.add_module("bn2", nn.BatchNorm2d(64))
        encoder.add_module("relu2", nn.ReLU())
        encoder.add_module("maxpool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        encoder.add_module("conv3", nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2))
        encoder.add_module("bn3", nn.BatchNorm2d(128))
        encoder.add_module("relu3", nn.ReLU())

        self.encoder = encoder
        self.fc = nn.Linear(8192, num_classes)
        if mlp:
            dim_mlp = self.fc.in_features
            # self.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.fc)
            self.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False),
                                        nn.BatchNorm1d(dim_mlp),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(dim_mlp, dim_mlp, bias=False),
                                        nn.BatchNorm1d(dim_mlp),
                                        nn.ReLU(inplace=True), # second layer
                                        self.fc)

    def forward(self, x):
        batch_size = x.size(0)
        feature = self.encoder(x)
        feature = feature.view(batch_size, 8192)
        feature = self.fc(feature)
        return feature

if __name__ == '__main__':
    net = CNN()
    y = net(torch.randn(512, 3, 32, 32))
    print(y.size())