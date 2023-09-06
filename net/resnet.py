'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from collections import OrderedDict
from typing import Tuple, Union

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

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
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


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    Code from CLIP
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """
    def __init__(self, block, layers, output_dim=10, mlp=False, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(block, width, layers[0])
        self.layer2 = self._make_layer(block, width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        heads = width//2
        attn_out_dim = 1024
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, attn_out_dim)
        
        self.fc = nn.Linear(attn_out_dim, output_dim)
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
    
    def _make_layer(self, block, planes, blocks, stride=1):
        layers = [block(self._inplanes, planes, stride)]

        self._inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        x = self.fc(x)

        return x


def ResNet18(num_classes=10, mlp=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, mlp)


def ResNet18_meta(num_classes=10, mlp=True):
    return ResNet_meta(BasicBlock_meta, [2, 2, 2, 2], num_classes, mlp)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes=10, mlp=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, mlp)


def ModifiedResNet50(num_classes=10, mlp=True):
    return ModifiedResNet(Bottleneck, [3, 4, 6, 3], num_classes, mlp)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())