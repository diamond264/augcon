'''In our digital experiments (Table 2),
our feature extractor is composed of three conv layers and
two fc layers. We present the conv layers as (input, output,
kernel, stride, padding) and fc layers as (input, output). The
three conv layers are: conv1 (3, 64, 5, 1, 2), conv2 (64, 64,
5, 1, 2), conv3 (64, 128, 5, 1, 2). The two fc layers are:
fc1 (8192, 3072), fc2 (3072, 2048). The architecture of the
feature generator is: (conv1, bn1, relu1, pool1)-(conv2, bn2,
relu2, pool2)-(conv3, bn3, relu3)-(fc1, bn4, relu4, dropout)-
(fc2, bn5, relu5). The classifier is a single fc layer, i.e. fc3
(2048, 10).'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        
        self.fc1 = nn.Linear(8192, 3072)
        self.bn4 = nn.BatchNorm1d(3072)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(3072, 2048)
        self.bn5 = nn.BatchNorm1d(2048)
        self.relu5 = nn.ReLU()
        
        self.fc3 = nn.Linear(2048, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = x.view(-1, 8192)
        x = self.dropout(self.relu4(self.bn4(self.fc1(x))))
        x = self.relu5(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x

def ResNet18(num_classes=10, mlp=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, mlp)


def ResNet18_meta(num_classes=10, mlp=True):
    return ResNet_meta(BasicBlock_meta, [2, 2, 2, 2], num_classes, mlp)