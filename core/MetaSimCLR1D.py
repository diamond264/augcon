import os
import random
import time
import sklearn.metrics as metrics

from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.models as models
import torch.multiprocessing as mp

# from datautils.SimCLR_dataset import subject_collate
from torch.utils.data import DataLoader, Dataset, DistributedSampler

class Encoder(nn.Module):
    def __init__(self, input_channels, z_dim):
        super(Encoder, self).__init__()
        self.vars = nn.ParameterList()

        self.num_blocks = 3
        in_dims = [input_channels, 32, 64]
        out_dims = [32, 64, z_dim]
        kernel_sizes = [24, 16, 8]
        
        for i in range(self.num_blocks):
            conv = nn.Conv1d(in_dims[i], out_dims[i], kernel_size=kernel_sizes[i])
            relu = nn.ReLU()
            dropout = nn.Dropout(0.1)
            
            w = nn.Parameter(torch.ones_like(conv.weight))
            torch.nn.init.kaiming_normal_(w)
            b = nn.Parameter(torch.zeros_like(conv.bias))
            self.vars.append(w)
            self.vars.append(b)

        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        
        idx = 0
        for i in range(self.num_blocks):
            w, b = vars[idx], vars[idx+1]
            idx += 2
            x = F.conv1d(x, w, b)
            x = F.relu(x, True)
            x = F.dropout(x, 0.1)
        
        x = F.adaptive_max_pool1d(x, 1)
        x = x.squeeze(-1)
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


class Classifier(nn.Module):
    def __init__(self, in_dim=96,
                 hidden_1=256, hidden_2=128, out_dim=50):
        super(Classifier, self).__init__()
        self.vars = nn.ParameterList()
        
        fc1 = nn.Linear(in_dim, hidden_1)
        relu1 = nn.ReLU()
        fc2 = nn.Linear(hidden_1, hidden_2)
        relu2 = nn.ReLU()
        fc3 = nn.Linear(hidden_2, out_dim)
        
        w = fc1.weight
        b = fc1.bias
        self.vars.append(w)
        self.vars.append(b)
        w = fc2.weight
        b = fc2.bias
        self.vars.append(w)
        self.vars.append(b)
        w = fc3.weight
        b = fc3.bias
        self.vars.append(w)
        self.vars.append(b)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        
        x = F.linear(x, vars[0], vars[1])
        x = F.relu(x, True)
        x = F.linear(x, vars[2], vars[3])
        x = F.relu(x, True)
        x = F.linear(x, vars[4], vars[5])
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


class SimCLRNet(nn.Module):
    """
    Build a SimCLR model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, input_channels=3, z_dim=96, out_dim=50, T=0.1, mlp=True):
        super(SimCLRNet, self).__init__()
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder = Encoder(input_channels, z_dim)

        self.mlp = mlp
        if mlp:  # hack: brute-force replacement
            self.classifier = Classifier(in_dim=z_dim, out_dim=out_dim)

    def forward(self, feature, aug_feature, vars=None):
        if vars is None:
            vars = nn.ParameterList()
            vars.extend(self.encoder.parameters())
            if self.mlp:
                vars.extend(self.classifier.parameters())
        
        enc_vars = vars[:len(self.encoder.parameters())]
        if self.mlp:
            cls_vars = vars[len(self.encoder.parameters()):]
            
        z = self.encoder(feature, enc_vars)  # queries: NxC
        aug_z = self.encoder(aug_feature, enc_vars)
        if self.mlp:
            z = self.classifier(z, cls_vars)
            aug_z = self.classifier(aug_z, cls_vars)
        
        z = F.normalize(z, dim=1)
        aug_z = F.normalize(aug_z, dim=1)
        
        LARGE_NUM = 1e9
        batch_size = z.size(0)

        labels = torch.arange(batch_size)
        masks = F.one_hot(torch.arange(batch_size), batch_size)

        logits_aa = torch.matmul(z, z.t())
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(aug_z, aug_z.t())
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(z, aug_z.t())

        logits = torch.cat([logits_ab, logits_aa, logits_bb], dim=1)
        logits /= self.T

        return logits, labels
    
    def adapt(self, feature, aug_feature, neg_feature, neg_aug_feature, full_batch_size, vars=None):
        if vars is None:
            vars = nn.ParameterList()
            vars.extend(self.encoder.parameters())
            if self.mlp:
                vars.extend(self.classifier.parameters())
        
        enc_vars = vars[:len(self.encoder.parameters())]
        if self.mlp:
            cls_vars = vars[len(self.encoder.parameters()):]
            
        z = self.encoder(feature, enc_vars)  # queries: NxC
        aug_z = self.encoder(aug_feature, enc_vars)
        neg_z = self.encoder(neg_feature, enc_vars)
        neg_aug_z = self.encoder(neg_aug_feature, enc_vars)
        if self.mlp:
            z = self.classifier(z, cls_vars)
            aug_z = self.classifier(aug_z, cls_vars)
            neg_z = self.classifier(neg_z, cls_vars)
            neg_aug_z = self.classifier(neg_aug_z, cls_vars)
        
        z = F.normalize(z, dim=1)
        aug_z = F.normalize(aug_z, dim=1)
        neg_z = F.normalize(neg_z, dim=1)
        neg_aug_z = F.normalize(neg_aug_z, dim=1)
        
        LARGE_NUM = 1e9
        
        pos_logits = torch.einsum('ij,ij->i', z, aug_z)
        pos_logits = torch.unsqueeze(pos_logits, dim=1)
        neg_logits_1 = torch.matmul(z, neg_z.t())
        neg_logits_2 = torch.matmul(z, neg_aug_z.t())
        neg_logits_3 = torch.matmul(aug_z, neg_aug_z.t())
        logits = torch.cat([pos_logits, neg_logits_1, neg_logits_2, neg_logits_3], dim=1)
        logits /= self.T
        logits = F.pad(logits, (0, full_batch_size*3-logits.shape[1]), "constant", -LARGE_NUM)
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        return logits, labels
    
    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                vars = nn.ParameterList()
                vars.extend(self.encoder.parameters())
                if self.mlp:
                    vars.extend(self.classifier.parameters())
            for p in vars:
                if p.grad is not None:
                    p.grad.zero_()
    
    def parameters(self):
        vars = nn.ParameterList()
        vars.extend(self.encoder.parameters())
        if self.mlp:
            vars.extend(self.classifier.parameters())
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


class SimCLRClassifier(nn.Module):
    def __init__(self, input_channels, z_dim, num_cls, mlp=True):
        super(SimCLRClassifier, self).__init__()
        self.base_model = Encoder(input_channels, z_dim)
        self.classifier = ClassificationHead(z_dim, z_dim, num_cls, mlp)
        
    def forward(self, x):
        x = self.base_model(x)
        pred = self.classifier(x)
        return pred


class MetaSimCLR1DLearner:
    def __init__(self, cfg, gpu, logger):
        self.cfg = cfg
        self.gpu = gpu
        
    def run(self, train_dataset, val_dataset, test_dataset):
        self.main_worker(0, 1, train_dataset, val_dataset, test_dataset)
    
    def main_worker(self, rank, world_size, train_dataset, val_dataset, test_dataset, logs=None):
        # Model initialization
        net = SimCLRNet(self.cfg.input_channels, self.cfg.z_dim, self.cfg.out_dim, self.cfg.T, True)
        if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
            cls_net = SimCLRClassifier(self.cfg.input_channels, self.cfg.z_dim, self.cfg.num_cls, self.cfg.mlp)
        
        meta_train_dataset = train_dataset
        
        if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
            # collate_fn = subject_collate if self.cfg.mode == 'pretrain' else None
            collate_fn = None
            train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size,
                                    shuffle=True, collate_fn=collate_fn,
                                    num_workers=self.cfg.num_workers, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size,
                                    shuffle=True, collate_fn=collate_fn,
                                    num_workers=self.cfg.num_workers, drop_last=True)
            if len(val_dataset) > 0:
                val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size,
                                        shuffle=True, collate_fn=collate_fn,
                                        num_workers=self.cfg.num_workers, drop_last=True)
        
        # Define criterion
        if self.cfg.criterion == 'crossentropy':
            criterion = nn.CrossEntropyLoss()
        
        # For finetuning, load pretrained model
        if self.cfg.mode == 'finetune':
            # Meta-train the pretrained model for domain adaptation
            if self.cfg.domain_adaptation:
                net.train()
                net.zero_grad()

                shuffled_idx = torch.randperm(len(meta_train_dataset))
                meta_train_dataset = torch.utils.data.Subset(meta_train_dataset, shuffled_idx)
                    
                support = [e[1] for e in meta_train_dataset]
                pos_support = [e[2] for e in meta_train_dataset]
                support = torch.stack(support, dim=0)
                pos_support = torch.stack(pos_support, dim=0)
                time.sleep(3)
                print("Perform domain adaptation step")
                start_time = time.time()
                enc_parameters = self.meta_train(rank, net, support, pos_support, criterion, log_steps=True, logs=logs)
                print("Time taken for meta-training: {}".format(time.time() - start_time))
                time.sleep(3)
            else:
                enc_parameters = list(net.parameters())
            
            print("Loading encoder parameters to the classifier")
            
            enc_dict = {}
            for idx, k in enumerate(list(net.state_dict().keys())):
                if not 'classifier' in k:
                    k_ = k.replace('encoder.', 'base_model.')
                    enc_dict[k_] = enc_parameters[idx]
            
            msg = cls_net.load_state_dict(enc_dict, strict=False)
            print("Missing keys: {}".format(msg.missing_keys))
            net = cls_net
            
            # Freezing the encoder
            if self.cfg.freeze:
                print("Freezing the encoder")
                for name, param in net.named_parameters():
                    if not 'classifier' in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
        
        parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
        if self.cfg.optimizer == 'sgd':
            optimizer = torch.optim.SGD(parameters, self.cfg.lr,
                                        momentum=self.cfg.momentum,
                                        weight_decay=self.cfg.wd)
        elif self.cfg.optimizer == 'adam':
            optimizer = torch.optim.Adam(parameters, self.cfg.lr,
                                        weight_decay=self.cfg.wd)
            
        # loss_best = 0
        if self.cfg.mode == 'finetune':
            time.sleep(3)
            print("Performing finetuning step")
            start_time = time.time()
            for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
                self.finetune(rank, net, train_loader, criterion, optimizer, epoch, self.cfg.epochs, logs)
            print("Time taken for finetuning: {}".format(time.time() - start_time))
            
        self.validate_finetune(rank, net, test_loader, criterion, logs)
    
    def meta_train(self, rank, net, support, pos_support, criterion, log_steps=False, logs=None):
        fast_weights = list(net.parameters())
        for i in range(self.cfg.task_steps):
            s_logits, s_targets = net(support, pos_support, fast_weights)
            s_loss = criterion(s_logits, s_targets)
            grad = torch.autograd.grad(s_loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.cfg.task_lr * p[0], zip(grad, fast_weights)))
        
        return fast_weights
    
    def finetune(self, rank, net, train_loader, criterion, optimizer, epoch, num_epochs, logs):
        net.eval()

        for batch_idx, data in enumerate(train_loader):
            features = data[0]
            targets = data[3]
            
            logits = net(features)
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
                
    def validate_finetune(self, rank, net, val_loader, criterion, logs):
        net.eval()
        
        total_targets = []
        total_logits = []
        total_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                features = data[0]
                targets = data[3]
                
                logits = net(features)
                
                total_loss += criterion(logits, targets)
                total_targets.append(targets)
                total_logits.append(logits)
            
            if len(total_targets) > 0:
                total_targets = torch.cat(total_targets, dim=0)
                total_logits = torch.cat(total_logits, dim=0)
                acc1, acc5 = self.accuracy(total_logits, total_targets, topk=(1, 5))
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
