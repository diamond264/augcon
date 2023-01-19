# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import core.resnet as resnet


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


class Encoder_res18(nn.Module):
    def __init__(self, avgpool=False, norm_layer=nn.BatchNorm2d):
        super(Encoder_res18, self).__init__()
        model = resnet.resnet18(norm_layer=norm_layer)
        # del(model.fc)
        model.fc = Identity()
        model.layer1 = Identity()
        model.layer2 = Identity()
        model.layer3 = Identity()
        model.layer4 = Identity()

        if not avgpool:
            # del(model.avgpool)
            model.avgpool = Identity()

        # print(model)
        self.model = model

    def forward(self, x):
        return self.model(x)

# class Encoder_simpleblock(nn.Module):
#     def __init__(self, channel=[1024, 512, 512], norm_layer=nn.BatchNorm2d):
#         super(Encoder_simpleblock, self).__init__()
#         self.blk1 = resnet.BasicBlock(channel[0], channel[1], 2, downsample)
#         self.blk2 = resnet.BasicBlock(channel[1], channel[2], 1)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#     def forward(self, x1, x2):
#         x = torch.cat((x1, x2), 1)
#         x = self.blk1(x)
#         x = self.blk2(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         return x

class Discriminator_res(nn.Module):
    def __init__(self, channel=[128, 256, 512], norm_layer=nn.BatchNorm2d):
        super(Discriminator_res, self).__init__()
        downsample=nn.Sequential(
                resnet.conv1x1(channel[0], channel[1], 2),
                norm_layer(channel[1]),
            )
        downsample2=nn.Sequential(
                resnet.conv1x1(channel[1], channel[2], 2),
                norm_layer(channel[2]),
            )

        self.blk1 = resnet.BasicBlock(channel[0], channel[1], 2, downsample)
        self.blk2 = resnet.BasicBlock(channel[1], channel[2], 2, downsample2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 64)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class AugCon(nn.Module):
    def __init__(self, encoder, discriminator, T=0.07):#, mlp=False):
        """
        T: softmax temperature (default: 0.07)
        """
        super(AugCon, self).__init__()

        self.T = T
        # create data encoder
        self.encoder = encoder
        # create relationship vector encoder
        self.discriminator = discriminator
        
        # if mlp:  # hack: brute-force replacement
        #     dim_mlp = self.encoder.fc.weight.shape[1]
        #     self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder.fc)
        #     self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder.fc)
        
    def forward(self, im_x1_a1, im_x1_a2, im_x2_a1, im_x2_a2):
        """
        Input:
            im_x1_a1: a batch of images
            im_x1_a2: a batch of augmented images
            im_x2_a1: another batch of images
            im_x2_a2: another batch of augmented images
        Output:
            logits, targets
        """
        # TODO: Issue - memory limitation
        # - Change SimCLR to MoCo
        # - Or utilize the idea of SupCon
        LARGE_NUM = 1e9
        # compute query features
        x1_a1 = self.encoder(im_x1_a1)
        x1_a2 = self.encoder(im_x1_a2)
        x1_rel = self.discriminator(x1_a1, x1_a2)
        
        x2_a1 = self.encoder(im_x2_a1)
        x2_a2 = self.encoder(im_x2_a2)
        x2_rel = self.discriminator(x2_a1, x2_a2)

        # HJ: After this part, I'm currently implementing the code
        l_pos = torch.einsum('nc,nc->n', [x1_rel, x2_rel]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [x1_rel, x1_rel.T])
        mask = torch.nn.functional.one_hot(torch.arange(x1_rel.shape[0]))*LARGE_NUM
        mask = mask.cuda()
        l_neg = l_neg - mask
        
        # concat calculated sample pairs
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return logits, labels