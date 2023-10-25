import os
import math
import time
import sklearn.metrics as metrics
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import defaultdict

# reference:
# https://github.com/facebookresearch/fairseq/blob/176cd934982212a4f75e0669ee81b834ee71dbb0/fairseq/models/wav2vec/wav2vec.py#L431
class Encoder(nn.Module):
    def __init__(self, input_channels=3, z_dim=256, num_blocks=4, kernel_sizes=[8, 4, 2, 1], strides=[4, 2, 1, 1]):
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks
        
        self.vars = nn.ParameterList()
        
        filters = [32, 64, 128, z_dim]
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        
        for i in range(num_blocks):
            block = nn.Sequential(nn.Conv1d(input_channels, filters[i],
                                            kernel_size=self.kernel_sizes[i],
                                            stride=self.strides[i]), 
                                  nn.ReLU(), 
                                  nn.Dropout(p=0.2))
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
            x = F.conv1d(x, w, b, self.strides[i])
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


class LeftZeroPad1d(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        
    def forward(self, x):
        # Compute the padding for the left side only
        padding = [self.k, 0]  # left, right
        x = F.pad(x, padding, "constant", 0)
        return x


class Aggregator(nn.Module):
    def __init__(self, num_blocks=5, num_filters=256, residual_scale=0.5):
        super(Aggregator, self).__init__()
        self.num_blocks = num_blocks
        
        self.vars = nn.ParameterList()
        self.zero_pads = []
        
        # define convolutional blocks with increasing kernel sizes
        kernel_sizes = [2, 3, 4, 5, 6, 7]
        if num_blocks == 9:
            kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3]
        for i in range(num_blocks):
            k = kernel_sizes[i]
            block = nn.Sequential(LeftZeroPad1d(k-1), 
                                  nn.Conv1d(num_filters, num_filters, kernel_size=k, dilation=1), 
                                  nn.GroupNorm(1, num_filters),
                                  nn.ReLU())
            # LeftZeroPad1d
            self.zero_pads.append(block[0])
            # Conv1d
            w = nn.Parameter(torch.ones_like(block[1].weight))
            torch.nn.init.kaiming_normal_(w)
            b = nn.Parameter(torch.zeros_like(block[1].bias))
            self.vars.append(w)
            self.vars.append(b)
            # GroupNorm
            w = nn.Parameter(torch.ones_like(block[2].weight))
            b = nn.Parameter(torch.zeros_like(block[2].bias))
            self.vars.append(w)
            self.vars.append(b)
            
        # define skip connections
        self.rproj = nn.ModuleList()
        for i in range(num_blocks):
            # rproj = nn.Conv1d(256, 256, kernel_size=1)
            rproj = None
            self.rproj.append(rproj)
        
        self.residual_scale = math.sqrt(residual_scale)
            
    def forward(self, z, vars=None):
        if vars is None:
            vars = self.vars
        
        idx = 0
        for i in range(self.num_blocks):
            residual = z
            z = self.zero_pads[i](z)
            
            w, b = vars[idx], vars[idx+1]
            idx += 2
            z = F.conv1d(z, w, b)
            
            w, b = vars[idx], vars[idx+1]
            idx += 2
            z = F.group_norm(z, 1, w, b)
            
            z = F.relu(z, True)
            
            rproj = self.rproj[i]
            if rproj != None:
                residual = rproj(residual)
            z = (z + residual) * self.residual_scale
        
        return z
    
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


class Predictor(nn.Module):
    def __init__(self, z_dim=256, pred_steps=12):
        super().__init__()
        self.vars = nn.ParameterList()
        
        self.pred_steps = pred_steps
        predictor = nn.ConvTranspose2d(
            z_dim, z_dim, (1, pred_steps)
        )
        w = nn.Parameter(torch.ones_like(predictor.weight))
        torch.nn.init.kaiming_normal_(w)
        b = nn.Parameter(torch.zeros_like(predictor.bias))
        self.vars.append(w)
        self.vars.append(b)
    
    def forward(self, z, vars=None):
        if vars is None:
            vars = self.vars
        
        z = F.conv_transpose2d(z, vars[0], vars[1])
        return z

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


class ClassificationHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_cls, mlp=False):
        super().__init__()
        if mlp:
            self.block = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_cls),
            )
        else:
            self.block = nn.Sequential(
                nn.Linear(input_size, num_cls),
            )
    
    def forward(self, x):
        x = self.block(x)
        return x


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]

class CPCNet(nn.Module):
    def __init__(self, input_channels=3, z_dim=256, enc_blocks=4, kernel_sizes=[8, 4, 2, 1], strides=[4, 2, 1, 1],
                 agg_blocks=4, pred_steps=12, n_negatives=15, offset=16):
        super(CPCNet, self).__init__()
        self.encoder = Encoder(input_channels, z_dim, enc_blocks, kernel_sizes, strides)
        self.aggregator = Aggregator(agg_blocks, z_dim)
        self.predictor = Predictor(z_dim, pred_steps)
        
        self.z_dim = z_dim
        self.pred_steps = pred_steps
        self.n_negatives = n_negatives
        self.offset = offset
        
        self.enc_param_idx = len(self.encoder.parameters())
        self.agg_param_idx = self.enc_param_idx+len(self.aggregator.parameters())
        self.pred_param_idx = self.agg_param_idx+len(self.predictor.parameters())

    def sample_negatives(self, y, n_negatives, bsz):
        _, fsz, tsz = y.shape
        cross_high = tsz * _

        y = y.transpose(0, 1)  # BCT -> CBT
        y = y.contiguous().view(fsz, -1)  # CBT => C(BxT)

        with torch.no_grad():
            # only perform cross-sample sampling
            if n_negatives > 0:
                tszs = (
                    buffered_arange(tsz)
                    .unsqueeze(-1)
                    .expand(-1, n_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, n_negatives * tsz),
                )
                for i, row in enumerate(cross_neg_idxs):
                    row[row >= tszs+i*bsz] += 1
                # cross_neg_idxs[cross_neg_idxs >= tszs] += 1
        neg_idxs = cross_neg_idxs

        negs = y[..., neg_idxs.view(-1)]
        negs = negs.view(
            fsz, bsz, n_negatives, tsz
        ).permute(
            2, 1, 0, 3
        )  # to NxBxCxT
        return negs
            
    def forward(self, x, vars=None, neg_x=None):
        if vars is None:
            vars = nn.ParameterList()
            vars.extend(self.encoder.parameters())
            vars.extend(self.aggregator.parameters())
            vars.extend(self.predictor.parameters())
        
        enc_vars = vars[:self.enc_param_idx]
        z = self.encoder(x, enc_vars)
        
        agg_vars = vars[self.enc_param_idx:self.agg_param_idx]
        z_hat = self.aggregator(z, agg_vars)
        z_hat = z_hat.unsqueeze(-1)
        
        pred_vars = vars[self.agg_param_idx:self.pred_param_idx]
        z_pred = self.predictor(z_hat, pred_vars)
        
        targets = z.unsqueeze(0)
        bsz = z.shape[0]
        if neg_x:
            neg_z = self.encoder(neg_x, enc_vars)
            neg_targets = self.sample_negatives(neg_z, self.n_negatives, bsz)
            targets = torch.cat([targets, neg_targets], dim=0)
        else:
            neg_targets = self.sample_negatives(z, self.n_negatives, bsz)
            targets = torch.cat([targets, neg_targets], dim=0)

        copies = targets.size(0)
        bsz, dim, tsz, steps = z_pred.shape
        steps = min(steps, tsz - self.offset)

        predictions = z_pred.new(
            bsz * copies * (tsz - self.offset + 1) * steps
            - ((steps + 1) * steps // 2) * copies * bsz
        )

        labels = predictions.new_full(
            (predictions.shape[0] // copies,), 0, dtype=torch.long
        )

        start = end = 0
        for i in range(steps):
            offset = i + self.offset
            end = start + (tsz - offset) * bsz * copies

            predictions[start:end] = torch.einsum(
                "bct,nbct->tbn", z_pred[..., :-offset, i], targets[..., offset:]
            ).flatten()

            start = end
        assert end == predictions.numel(), "{} != {}".format(end, predictions.numel())

        temp = 1
        predictions = predictions.view(-1, copies)/temp

        return predictions, labels
    
    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                vars = nn.ParameterList()
                vars.extend(self.encoder.parameters())
                vars.extend(self.aggregator.parameters())
                vars.extend(self.predictor.parameters())
            for p in vars:
                if p.grad is not None:
                    p.grad.zero_()

    def parameters(self):
        vars = nn.ParameterList()
        vars.extend(self.encoder.parameters())
        vars.extend(self.aggregator.parameters())
        vars.extend(self.predictor.parameters())
        return vars


class CPCClassifier(nn.Module):
    def __init__(self, input_channels=3, z_dim=256, enc_blocks=4, kernel_sizes=[8, 4, 2, 1], strides=[4, 2, 1, 1],
                 agg_blocks=4, pooling='mean', num_cls=10, mlp=False):
        super(CPCClassifier, self).__init__()
        self.encoder = Encoder(input_channels, z_dim, enc_blocks, kernel_sizes, strides)
        self.aggregator = Aggregator(agg_blocks, z_dim)
        self.pooling = pooling
        self.classifier = ClassificationHead(z_dim, z_dim, num_cls, mlp)
            
    def forward(self, x):
        x = self.encoder(x)
        x = self.aggregator(x)
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


class MetaCPCLearner:
    def __init__(self, cfg, gpu, logger):
        self.cfg = cfg
        self.gpu = gpu
    
    def run(self, train_dataset, val_dataset, test_dataset):
        self.main_worker(0, 1, train_dataset, val_dataset, test_dataset)
        
    def main_worker(self, rank, world_size, train_dataset, val_dataset, test_dataset, logs=None):
        # Model initialization
        net = CPCNet(self.cfg.input_channels, self.cfg.z_dim, self.cfg.enc_blocks, self.cfg.kernel_sizes, self.cfg.strides,
                         self.cfg.agg_blocks, self.cfg.pred_steps, self.cfg.n_negatives, self.cfg.offset)
        if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval':
            cls_net = CPCClassifier(self.cfg.input_channels, self.cfg.z_dim, self.cfg.enc_blocks, self.cfg.kernel_sizes, self.cfg.strides,
                                    self.cfg.agg_blocks, self.cfg.pooling, self.cfg.num_cls, self.cfg.mlp)
        
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
        if self.cfg.criterion == 'crossentropy':
            criterion = nn.CrossEntropyLoss()
        
        if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval':
            # Freeze the encoder part of the network
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
            
            # Meta-train the pretrained model for domain adaptation
            if self.cfg.domain_adaptation:
                net.zero_grad()
                support = [e[0] for e in meta_train_dataset]
                support = torch.stack(support, dim=0)
                # sleep for 5 seconds
                time.sleep(3)
                print("Perform domain adaptation step")
                start_time = time.time()
                enc_parameters = self.meta_train(rank, net, support, criterion, log_internals=True, logs=logs)
                # time in seconds with two floating points
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
            
            if self.cfg.mode == 'finetune':
                time.sleep(3)
                print("Performing finetuning step")
                start_time = time.time()
                for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
                    # print("ft epoch: {}".format(epoch))
                    self.finetune(rank, cls_net, train_loader, criterion, optimizer, epoch, self.cfg.epochs, logs)
                    # if len(val_dataset) > 0:
                    #     self.validate(rank, cls_net, val_loader, criterion, logs)
                print("Time taken for finetuning: {}".format(time.time() - start_time))
            
            self.validate(rank, cls_net, test_loader, criterion, logs)
    
    def finetune(self, rank, net, train_loader, criterion, optimizer, epoch, num_epochs, logs):
        net.eval()
        
        for batch_idx, data in enumerate(train_loader):
            features = data[0]
            targets = data[1]
            
            logits = net(features)
            loss = criterion(logits, targets)
            
            # if rank == 0:
            #     if batch_idx % self.cfg.log_freq == 0:
            #         acc1, acc3 = self.accuracy(logits, targets, topk=(1, 3))
            #         log = f'Epoch [{epoch+1}/{num_epochs}]-({batch_idx}/{len(train_loader)}) '
            #         log += f'Loss: {loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(3): {acc3.item():.2f}'
            #         print(log)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def meta_train(self, rank, net, support, criterion, log_internals=False, logs=None):
        fast_weights = list(net.parameters())
        for i in range(self.cfg.task_steps):
            s_logits, s_targets = net(support, fast_weights)
            s_loss = criterion(s_logits, s_targets)
            grad = torch.autograd.grad(s_loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.cfg.task_lr * p[0], zip(grad, fast_weights)))
            
            # if log_internals and rank == 0:
            #     acc1, acc3 = self.accuracy(s_logits, s_targets, topk=(1, 3))
            #     print(f'\tmeta-train [{i}/{self.cfg.task_steps}] Loss: {s_loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(3): {acc3.item():.2f}')
        
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
        
        log = f'\tValidation Loss: {total_loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(3): {acc3.item():.2f}'
        log += f', F1: {f1.item():.2f}, Recall: {recall.item():.2f}, Precision: {precision.item():.2f}'
        print(log)
                
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