'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class BasicBlock_meta(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_meta, self).__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.vars = nn.ParameterList()
        
        conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        w = nn.Parameter(torch.ones_like(conv1.weight))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        
        bn1 = nn.BatchNorm2d(planes)
        w = nn.Parameter(torch.ones_like(bn1.weight))
        b = nn.Parameter(torch.zeros_like(bn1.bias))
        running_mean = nn.Parameter(torch.zeros_like(bn1.running_mean), requires_grad=False)
        running_var = nn.Parameter(torch.ones_like(bn1.running_var), requires_grad=False)
        num_batches_tracked = nn.Parameter(torch.zeros_like(bn1.num_batches_tracked), requires_grad=False)
        self.vars.extend([w, b, running_mean, running_var, num_batches_tracked])
        
        conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        w = nn.Parameter(torch.ones_like(conv2.weight))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        
        bn2 = nn.BatchNorm2d(planes)
        w = nn.Parameter(torch.ones_like(bn2.weight))
        b = nn.Parameter(torch.zeros_like(bn2.bias))
        running_mean = nn.Parameter(torch.zeros_like(bn2.running_mean), requires_grad=False)
        running_var = nn.Parameter(torch.ones_like(bn2.running_var), requires_grad=False)
        num_batches_tracked = nn.Parameter(torch.zeros_like(bn2.num_batches_tracked), requires_grad=False)
        self.vars.extend([w, b, running_mean, running_var, num_batches_tracked])
        
        if stride != 1 or in_planes != self.expansion*planes:
            downsample_conv = nn.Conv2d(in_planes, self.expansion*planes,
                                        kernel_size=1, stride=stride, bias=False)
            w = nn.Parameter(torch.ones_like(downsample_conv.weight))
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            
            downsample_bn = nn.BatchNorm2d(self.expansion*planes)
            w = nn.Parameter(torch.ones_like(downsample_bn.weight))
            b = nn.Parameter(torch.zeros_like(downsample_bn.bias))
            running_mean = nn.Parameter(torch.zeros_like(downsample_bn.running_mean), requires_grad=False)
            running_var = nn.Parameter(torch.ones_like(downsample_bn.running_var), requires_grad=False)
            num_batches_tracked = nn.Parameter(torch.zeros_like(downsample_bn.num_batches_tracked), requires_grad=False)
            self.vars.extend([w, b, running_mean, running_var, num_batches_tracked])
            
    def forward(self, x, vars=None, bn_training=True):
        if vars is None:
            vars = self.vars
        
        out = F.conv2d(x, vars[0], stride=self.stride, padding=1)
        out = F.batch_norm(out, vars[3], vars[4], vars[1], vars[2], bn_training)
        out = F.relu(out)
        
        out = F.conv2d(out, vars[6], stride=1, padding=1)
        out = F.batch_norm(out, vars[9], vars[10], vars[7], vars[8], bn_training)
        
        downsampled = x
        if self.stride != 1 or self.in_planes != self.expansion*self.planes:
            downsampled = F.conv2d(x, vars[12], stride=self.stride)
            downsampled = F.batch_norm(downsampled, vars[15], vars[16], vars[13], vars[14], bn_training)
        
        out += downsampled
        out = F.relu(out)
        return out
    
    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()
    
    def parameters(self):
        return self.vars


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, mlp=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
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

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet_meta(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, mlp=True):
        super(ResNet_meta, self).__init__()
        in_planes = 64
        self.block = block
        self.num_blocks = num_blocks
        self.mlp = mlp
        self.vars = nn.ParameterList()
        
        conv1 = nn.Conv2d(3, 64, kernel_size=7,
                          stride=2, padding=3, bias=False)
        w = nn.Parameter(torch.ones_like(conv1.weight))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        
        bn1 = nn.BatchNorm2d(64)
        w = nn.Parameter(torch.ones_like(bn1.weight))
        b = nn.Parameter(torch.zeros_like(bn1.bias))
        running_mean = nn.Parameter(torch.zeros_like(bn1.running_mean), requires_grad=False)
        running_var = nn.Parameter(torch.ones_like(bn1.running_var), requires_grad=False)
        num_batches_tracked = nn.Parameter(torch.zeros_like(bn1.num_batches_tracked), requires_grad=False)
        self.vars.extend([w, b, running_mean, running_var, num_batches_tracked])
        
        layer1, in_planes = self._make_layer(block, 64, num_blocks[0], stride=1, in_planes=in_planes)
        self.vars.extend(self.get_layer_params(layer1))
        layer2, in_planes = self._make_layer(block, 128, num_blocks[1], stride=2, in_planes=in_planes)
        self.vars.extend(self.get_layer_params(layer2))
        layer3, in_planes = self._make_layer(block, 256, num_blocks[2], stride=2, in_planes=in_planes)
        self.vars.extend(self.get_layer_params(layer3))
        layer4, in_planes = self._make_layer(block, 512, num_blocks[3], stride=2, in_planes=in_planes)
        self.vars.extend(self.get_layer_params(layer4))
        
        fc = nn.Linear(512*block.expansion, num_classes)
        if mlp:
            dim_mlp = fc.in_features
            fc2 = nn.Linear(dim_mlp, dim_mlp, bias=False)
            w = nn.Parameter(torch.ones_like(fc2.weight))
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            
            bn1 = nn.BatchNorm1d(dim_mlp)
            w = nn.Parameter(torch.ones_like(bn1.weight))
            b = nn.Parameter(torch.zeros_like(bn1.bias))
            running_mean = nn.Parameter(torch.zeros_like(bn1.running_mean), requires_grad=False)
            running_var = nn.Parameter(torch.ones_like(bn1.running_var), requires_grad=False)
            num_batches_tracked = nn.Parameter(torch.zeros_like(bn1.num_batches_tracked), requires_grad=False)
            self.vars.extend([w, b, running_mean, running_var, num_batches_tracked])
            
            fc3 = nn.Linear(dim_mlp, dim_mlp, bias=False)
            w = nn.Parameter(torch.ones_like(fc3.weight))
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            
            bn2 = nn.BatchNorm1d(dim_mlp)
            w = nn.Parameter(torch.ones_like(bn2.weight))
            b = nn.Parameter(torch.zeros_like(bn2.bias))
            running_mean = nn.Parameter(torch.zeros_like(bn2.running_mean), requires_grad=False)
            running_var = nn.Parameter(torch.ones_like(bn2.running_var), requires_grad=False)
            num_batches_tracked = nn.Parameter(torch.zeros_like(bn2.num_batches_tracked), requires_grad=False)
            self.vars.extend([w, b, running_mean, running_var, num_batches_tracked])
            
        w = nn.Parameter(torch.ones_like(fc.weight))
        b = nn.Parameter(torch.zeros_like(fc.bias))
        torch.nn.init.kaiming_normal_(w)
        self.vars.extend([w, b])

    def _make_layer(self, block, planes, num_blocks, stride, in_planes):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, planes, stride))
            in_planes = planes * block.expansion
        return nn.Sequential(*layers), in_planes
    
    def get_layer_params(self, layers):
        vars = nn.ParameterList()
        for layer in layers:
            vars.extend(layer.parameters())
        return vars
    
    def forward_layers(self, layers, x, vars, var_idx):
        for layer in layers:
            parsed_vars = vars[var_idx:var_idx+len(layer.parameters())]
            x = layer(x, parsed_vars)
            var_idx += len(layer.parameters())
        return x, var_idx

    def forward(self, x, vars=None, bn_training=True):
        if vars is None:
            vars = self.vars
            
        x = F.conv2d(x, vars[0], stride=2, padding=3)
        x = F.batch_norm(x, vars[3], vars[4], vars[1], vars[2], training=bn_training)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        var_idx = 6
        
        in_planes = 64
        layer1, in_planes = self._make_layer(self.block, 64, self.num_blocks[0], stride=1, in_planes=in_planes)
        layer2, in_planes = self._make_layer(self.block, 128, self.num_blocks[1], stride=2, in_planes=in_planes)
        layer3, in_planes = self._make_layer(self.block, 256, self.num_blocks[2], stride=2, in_planes=in_planes)
        layer4, in_planes = self._make_layer(self.block, 512, self.num_blocks[3], stride=2, in_planes=in_planes)
        x, var_idx = self.forward_layers(layer1, x, vars, var_idx)
        x, var_idx = self.forward_layers(layer2, x, vars, var_idx)
        x, var_idx = self.forward_layers(layer3, x, vars, var_idx)
        x, var_idx = self.forward_layers(layer4, x, vars, var_idx)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        if self.mlp:
            x = F.linear(x, vars[var_idx])
            var_idx += 1
            x = F.batch_norm(x, vars[var_idx+2], vars[var_idx+3], vars[var_idx], vars[var_idx+1], training=bn_training)
            var_idx += 5
            x = F.relu(x)
            x = F.linear(x, vars[var_idx])
            var_idx += 1
            x = F.batch_norm(x, vars[var_idx+2], vars[var_idx+3], vars[var_idx], vars[var_idx+1], training=bn_training)
            var_idx += 5
            x = F.relu(x)
        
        x = F.linear(x, vars[var_idx], vars[var_idx+1])
        return x
    
    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()
    
    def parameters(self):
        return self.vars


def ResNet18(num_classes=10, mlp=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, mlp)


def ResNet18_meta(num_classes=10, mlp=True):
    return ResNet_meta(BasicBlock_meta, [2, 2, 2, 2], num_classes, mlp)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())