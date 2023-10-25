import os
import math
import time
import sklearn.metrics as metrics
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from collections import defaultdict

class Encoder(nn.Module):
    def __init__(self, input_channels=3, z_dim=256, num_blocks=3, kernel_sizes=[3, 3, 3, 1], strides=[2, 1, 1, 1]):
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks
        
        self.vars = nn.ParameterList()
        
        filters = [32, 64, 128, z_dim]
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        
        for i in range(num_blocks):
            block = nn.Sequential(nn.Conv1d(input_channels, filters[i],
                                            kernel_size=self.kernel_sizes[i])#,
                                            # stride=self.strides[i]), 
                                  )
            input_channels = filters[i]
            
            w = nn.Parameter(torch.ones_like(block[0].weight))
            torch.nn.init.kaiming_normal_(w)
            b = nn.Parameter(torch.zeros_like(block[0].bias))
            self.vars.append(w)
            self.vars.append(b)
    
    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        
        idx = 0
        for i in range(self.num_blocks):
            w, b = vars[idx], vars[idx+1]
            idx += 2
            x = F.conv1d(x, w, b)#, self.strides[i])
            x = F.relu(x, True)
            x = F.dropout(x, 0.2)
            
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


class Decoder(nn.Module):
    def __init__(self, input_channels=3, z_dim=128, num_blocks=3, kernel_sizes=[3, 3, 3, 4], strides=[1, 1, 1, 2]):
        super(Decoder, self).__init__()
        self.num_blocks = num_blocks
        
        self.vars = nn.ParameterList()
        
        filters = [64, 32, input_channels]
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        
        for i in range(num_blocks):
            if i == 3: padding = 0
            else: padding = 0
            if i == 3: activation = nn.Tanh()
            else: activation = nn.ReLU()
            if i != 3: dropout = nn.Dropout(p=0.2)
            else: dropout = nn.Dropout(p=0)
            block = nn.Sequential(nn.ConvTranspose1d(z_dim, filters[i],
                                            kernel_size=self.kernel_sizes[i])#,
                                            # stride=self.strides[i],
                                            # output_padding=padding), 
                                  )
            z_dim = filters[i]
            
            w = nn.Parameter(torch.ones_like(block[0].weight))
            torch.nn.init.kaiming_normal_(w)
            b = nn.Parameter(torch.zeros_like(block[0].bias))
            self.vars.append(w)
            self.vars.append(b)
    
    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        
        idx = 0
        for i in range(self.num_blocks):
            if i == 3: padding = 0
            else: padding = 0
            w, b = vars[idx], vars[idx+1]
            idx += 2
            x = F.conv_transpose1d(x, w, b)#, self.strides[i], output_padding=padding)
            if i == 2: x = F.sigmoid(x)
            else: x = F.relu(x, True)
            if i!= 2: x = F.dropout(x, 0)
            else: x = F.dropout(x, 0)
            
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


class AutoEncoderNet(nn.Module):
    """
    Build a SimCLR model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, input_channels=3, z_dim=96):
        super(AutoEncoderNet, self).__init__()

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder = Encoder(input_channels, z_dim)
        self.decoder = Decoder(input_channels, z_dim)

    def forward(self, x, vars=None):
        if vars is None:
            vars = nn.ParameterList()
            vars.extend(self.encoder.parameters())
            vars.extend(self.decoder.parameters())
        
        enc_vars = vars[:len(self.encoder.parameters())]
        dec_vars = vars[len(self.encoder.parameters()):]
        
        z = self.encoder(x, enc_vars)
        x_hat = self.decoder(z, dec_vars)
        return x_hat
    
    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                self.encoder.zero_grad()
                self.decoder.zero_grad()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()
    
    def parameters(self):
        vars = nn.ParameterList()
        vars.extend(self.encoder.parameters())
        vars.extend(self.decoder.parameters())
        return vars


class ClassificationHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_cls, mlp=False):
        super().__init__()
        if mlp:
            self.block = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(128, num_cls),
            )
        else:
            self.block = nn.Sequential(
                nn.Linear(input_size, num_cls),
            )
    
    def forward(self, x):
        x = self.block(x)
        return x


class AutoEncoderClassifier(nn.Module):
    def __init__(self, input_channels, z_dim, num_cls, mlp=True):
        super(AutoEncoderClassifier, self).__init__()
        self.base_model = Encoder(input_channels, z_dim)
        self.classifier = ClassificationHead(z_dim, z_dim, num_cls, mlp)
        self.pooling = 'mean'
            
    def forward(self, x):
        x = self.base_model(x)
        # x = self.aggregator(x)
        if self.pooling == 'mean':
            c = torch.mean(x, dim=2)
        elif self.pooling == 'max':
            c, _ = torch.max(x, dim=2)
        elif self.pooling == 'sum':
            c = torch.sum(x, dim=2)
        else:
            raise ValueError("Invalid pooling mode. Please choose from 'mean', 'max', or 'sum'.")
        pred = self.classifier(c)
        return pred


class MetaAutoEncoderLearner:
    def __init__(self, cfg, gpu, logger):
        self.cfg = cfg
        self.gpu = gpu
        self.logger = logger
        
    def run(self, train_dataset, val_dataset, test_dataset):
        num_gpus = len(self.gpu)
        logs = mp.Manager().list([])
        self.logger.info("Executing SimCLR")
        self.logger.info("Logs are skipped during training")
        if num_gpus > 1:
            mp.spawn(self.main_worker,
                     args=(num_gpus, train_dataset, val_dataset, test_dataset, logs),
                     nprocs=num_gpus)
        else:
            self.main_worker(0, 1, train_dataset, val_dataset, test_dataset, logs)
        
        for log in logs:
            self.logger.info(log)
    
    def main_worker(self, rank, world_size, train_dataset, val_dataset, test_dataset, logs):
        # Model initialization
        net = AutoEncoderNet(self.cfg.input_channels, self.cfg.z_dim)
        if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
            cls_net = AutoEncoderClassifier(self.cfg.input_channels, self.cfg.z_dim, self.cfg.num_cls, self.cfg.mlp)
            
        if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval':
            train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size,
                                    shuffle=True, num_workers=self.cfg.num_workers, drop_last=True)
            if len(val_dataset) > 0:
                val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size,
                                        shuffle=True, num_workers=self.cfg.num_workers, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size,
                                    shuffle=True, num_workers=self.cfg.num_workers, drop_last=True)
        meta_train_dataset = train_dataset
        
        # Define criterion
        if self.cfg.criterion == 'mse':
            criterion = nn.MSELoss()
        elif self.cfg.criterion == 'crossentropy':
            criterion = nn.CrossEntropyLoss()
        
        # For finetuning, load pretrained model
        if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval':
            # Meta-train the pretrained model for domain adaptation
            if self.cfg.domain_adaptation:
                da_criterion = nn.MSELoss()
                # net.train()
                net.zero_grad()
                support = [e[0] for e in meta_train_dataset]
                support = torch.stack(support, dim=0)
                time.sleep(3)
                print("Perform domain adaptation step")
                start_time = time.time()
                enc_parameters = self.meta_train(rank, net, support, da_criterion, log_internals=True, logs=logs)
                print("Time taken for meta-training: {}".format(time.time() - start_time))
                time.sleep(3)
            else:
                enc_parameters = list(net.parameters())
            print("Loading encoder parameters to the classifier")
            
            enc_dict = {}
            for idx, k in enumerate(list(cls_net.state_dict().keys())):
                if not 'classifier' in k:
                    enc_dict[k] = enc_parameters[idx]
            
            msg = cls_net.load_state_dict(enc_dict, strict=False)
            print("Missing keys: {}".format(msg.missing_keys))
            
            if self.cfg.freeze:
                for name, param in cls_net.named_parameters():
                    if not 'classifier' in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            
            # Defining optimizer for classifier
            parameters = list(filter(lambda p: p.requires_grad, cls_net.parameters()))
            if self.cfg.optimizer == 'sgd':
                optimizer = torch.optim.SGD(parameters, self.cfg.lr,
                                            momentum=self.cfg.momentum,
                                            weight_decay=self.cfg.wd)
            elif self.cfg.optimizer == 'adam':
                optimizer = torch.optim.Adam(parameters, self.cfg.lr,
                                            weight_decay=self.cfg.wd)
            
            if self.cfg.mode == 'finetune':
                time.sleep(3)
                print("Performing finetuning step")
                start_time = time.time()
                for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
                    self.finetune(rank, cls_net, train_loader, criterion, optimizer, epoch, self.cfg.epochs, logs)
                print("Time taken for finetuning: {}".format(time.time() - start_time))
            self.validate(rank, cls_net, test_loader, criterion, logs)
    
    def finetune(self, rank, net, train_loader, criterion, optimizer, epoch, num_epochs, logs):
        net.eval()
        
        for batch_idx, data in enumerate(train_loader):
            features = data[0]
            targets = data[1]
            
            logits = net(features)
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def meta_train(self, rank, net, support, criterion, log_internals=False, logs=None):
        fast_weights = list(net.parameters())
        for i in range(self.cfg.task_steps):
            support_hat = net(support, fast_weights)
            s_loss = criterion(support, support_hat)
            grad = torch.autograd.grad(s_loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.cfg.task_lr * p[0], zip(grad, fast_weights)))
        
        return fast_weights
    
    def validate(self, rank, net, val_loader, criterion, logs):
        net.eval()
        
        total_targets = []
        total_logits = []
        total_loss = 0
        for batch_idx, data in enumerate(val_loader):
            features = data[0]
            targets = data[1]
            
            logits = net(features)
            total_loss += criterion(logits, targets)
            total_targets.append(targets)
            total_logits.append(logits)
        
        total_targets = torch.cat(total_targets, dim=0)
        total_logits = torch.cat(total_logits, dim=0)
        acc1, acc3 = self.accuracy(total_logits, total_targets, topk=(1, 3))
        f1, recall, precision = self.scores(total_logits, total_targets)
        total_loss /= len(val_loader)
                
    def accuracy(self, output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    
    def scores(self, output, target):
        with torch.no_grad():
            out_val = torch.flatten(torch.argmax(output, dim=1)).cpu().numpy()
            target_val = torch.flatten(target).cpu().numpy()
            
            cohen_kappa = metrics.cohen_kappa_score(target_val, out_val)
            precision = metrics.precision_score(
                target_val, out_val, average="macro", zero_division=0
            )
            recall = metrics.recall_score(
                target_val, out_val, average="macro", zero_division=0
            )
            f1 = metrics.f1_score(
                target_val, out_val, average="macro", zero_division=0
            )
            acc = metrics.accuracy_score(
                target_val, out_val
            )

            return f1, recall, precision