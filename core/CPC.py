import os
import math
import sklearn.metrics as metrics

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.optim.lr_scheduler import StepLR

# reference:
# https://github.com/facebookresearch/fairseq/blob/176cd934982212a4f75e0669ee81b834ee71dbb0/fairseq/models/wav2vec/wav2vec.py#L431

class Encoder(nn.Module):
    def __init__(self, input_channels=3, z_dim=256, num_blocks=4, kernel_sizes=[8, 4, 2, 1], strides=[4, 2, 1, 1]):
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks
        
        filters = [32, 64, 128, z_dim]
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.Sequential(nn.Conv1d(input_channels, filters[i],
                                            kernel_size=self.kernel_sizes[i],
                                            stride=self.strides[i]), 
                                  nn.ReLU(), 
                                  nn.Dropout(p=0.2))
            self.blocks.append(block)
            input_channels = filters[i]
    
    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.blocks[i](x)
        return x


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
        # define convolutional blocks with increasing kernel sizes
        kernel_sizes = [2, 3, 4, 5, 6, 7]
        if num_blocks == 9:
            kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3]
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            k = kernel_sizes[i]
            zero_pad = LeftZeroPad1d(k-1)
            conv_layer = nn.Conv1d(num_filters, num_filters, kernel_size=k, dilation=1)
            norm_layer = nn.GroupNorm(1, num_filters)
            relu_layer = nn.ReLU()
            self.blocks.append(nn.Sequential(zero_pad, conv_layer, norm_layer, relu_layer))
            
        # define skip connections
        self.rproj = nn.ModuleList()
        for i in range(num_blocks):
            # rproj = nn.Conv1d(256, 256, kernel_size=1)
            rproj = None
            self.rproj.append(rproj)
        
        self.residual_scale = math.sqrt(residual_scale)
            
    def forward(self, z):
        # perform autoregressive summarization with convolutional blocks
        block_outputs = []
        for i in range(self.num_blocks):
            residual = z
            for layer in self.blocks[i]:
                z = layer(z)
            
            rproj = self.rproj[i]
            if rproj != None:
                residual = rproj(residual)
            z = (z + residual) * self.residual_scale
        return z


class Predictor(nn.Module):
    def __init__(self, z_dim=256, pred_steps=12):
        super().__init__()
        self.predictor = nn.ConvTranspose2d(
            z_dim, z_dim, (1, pred_steps)
        )
    
    def forward(self, z):
        z = self.predictor(z)
        return z


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
            
    def forward(self, x, neg_x=None):
        z = self.encoder(x)
        z_hat = self.aggregator(z)
        z_hat = z_hat.unsqueeze(-1)
        z_pred = self.predictor(z_hat)
        
        targets = z.unsqueeze(0)
        bsz = z.shape[0]
        if neg_x:
            neg_z = self.encoder(neg_x)
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


class CPCLearner:
    def __init__(self, cfg, gpu, logger):
        self.cfg = cfg
        self.gpu = gpu
        self.logger = logger
    
    def write_log(self, rank, logs, log):
        if rank == 0:
            logs.append(log)
            print(log)
    
    def run(self, train_dataset, val_dataset, test_dataset):
        num_gpus = len(self.gpu)
        logs = mp.Manager().list([])
        self.logger.info("Executing CPC")
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
        torch.cuda.set_device(rank)
        
        # Model initialization
        if self.cfg.mode == 'pretrain':
            net = CPCNet(self.cfg.input_channels, self.cfg.z_dim, self.cfg.enc_blocks, self.cfg.kernel_sizes, self.cfg.strides,
                         self.cfg.agg_blocks, self.cfg.pred_steps, self.cfg.n_negatives, self.cfg.offset)
        elif self.cfg.mode == 'finetune' or self.cfg.mode == 'eval':
            net = CPCClassifier(self.cfg.input_channels, self.cfg.z_dim, self.cfg.enc_blocks, self.cfg.kernel_sizes, self.cfg.strides,
                                self.cfg.agg_blocks, self.cfg.pooling, self.cfg.num_cls, self.cfg.mlp)
        net.cuda()
        
        # DDP setting
        if world_size > 1:
            dist.init_process_group(backend='nccl',
                                    init_method=self.cfg.dist_url,
                                    world_size=world_size,
                                    rank=rank)
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = nn.parallel.DistributedDataParallel(net, device_ids=[rank], find_unused_parameters=True)
            
            train_sampler = DistributedSampler(train_dataset)
            if len(val_dataset) > 0:
                val_sampler = DistributedSampler(val_dataset)
            test_sampler = DistributedSampler(test_dataset)
            train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size,#//world_size,
                                      shuffle=False, sampler=train_sampler, num_workers=self.cfg.num_workers, drop_last=True)
            if len(val_dataset) > 0:
                val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size,#//world_size,
                                        shuffle=False, sampler=val_sampler, num_workers=self.cfg.num_workers, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size,#//world_size,
                                      shuffle=False, sampler=test_sampler, num_workers=self.cfg.num_workers, drop_last=True)
            self.write_log(rank, logs, "DDP is used for training - training {} instances for each worker".format(len(list(train_sampler))))
        # Single GPU setting
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size,
                                      shuffle=True, num_workers=self.cfg.num_workers, drop_last=True)
            if len(val_dataset) > 0:
                val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size,
                                        shuffle=True, num_workers=self.cfg.num_workers, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size,
                                      shuffle=True, num_workers=self.cfg.num_workers, drop_last=True)
            self.write_log(rank, logs, "Single GPU is used for training - training {} instances for each worker".format(len(train_dataset)))
        
        # self.all_domains = self.split_per_domain(train_dataset)
        
        # Define criterion
        if self.cfg.criterion == 'crossentropy':
            criterion = nn.CrossEntropyLoss().cuda()
        
        # For finetuning and evaluation, load pretrained model
        if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval':
            if os.path.isfile(self.cfg.pretrained):
                self.write_log(rank, logs, "Loading pretrained model from checkpoint - {}".format(self.cfg.pretrained))
                loc = 'cuda:{}'.format(rank)
                state = torch.load(self.cfg.pretrained, map_location=loc)['state_dict']
                
                for k, v in list(state.items()):
                    if world_size > 1:
                        k = 'module.' + k
                    if k in net.state_dict().keys():
                        state[k] = v
                
                msg = net.load_state_dict(state, strict=False)
                self.write_log(rank, logs, f"missing keys: {msg.missing_keys}")
            else:
                if rank == 0:
                    log = "No checkpoint found at '{}'".format(self.cfg.pretrained)
                    logs.append(log)
                    print(log)
            
            # Freezing the encoder
            if self.cfg.freeze:
                if rank == 0:
                    log = "Freezing the encoder"
                    logs.append(log)
                    print(log)
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
        
        # Load checkpoint if exists
        if os.path.isfile(self.cfg.resume):
            if rank == 0:
                log = "Loading state_dict from checkpoint - {}".format(self.cfg.resume)
                logs.append(log)
                print(log)
            loc = 'cuda:{}'.format(rank)
            state = torch.load(self.cfg.resume, map_location=loc)
            
            for k, v in list(state['state_dict'].items()):
                if world_size > 1:
                    k = 'module.' + k
                if k in net.state_dict().keys():
                    state['state_dict'][k] = v
            
            net.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            self.cfg.start_epoch = state['epoch']
        
        # scheduler = StepLR(optimizer, step_size=self.cfg.lr_decay_step, gamma=self.cfg.lr_decay)
        
        if self.cfg.mode != 'eval':
            loss_best = 0
            for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
                if world_size > 1:
                    train_sampler.set_epoch(epoch)
                    if len(val_dataset) > 0:
                        val_sampler.set_epoch(epoch)
                    test_sampler.set_epoch(epoch)
                
                if self.cfg.mode == 'pretrain':
                    self.pretrain(rank, net, train_loader, criterion, optimizer, epoch, self.cfg.epochs, logs)
                    if len(val_dataset) > 0:
                        loss_ep = self.validate_pretrain(rank, net, val_loader, criterion, logs)
                        
                        if loss_best == 0 or loss_ep < loss_best:
                            loss_best = loss_ep
                            esnum = 0
                            if rank == 0:
                                ckpt_dir = self.cfg.ckpt_dir
                                ckpt_filename = 'checkpoint_best.pth.tar'
                                ckpt_filename = os.path.join(ckpt_dir, ckpt_filename)
                                state_dict = net.state_dict()
                                if world_size > 1:
                                    for k, v in list(state_dict.items()):
                                        if 'module.' in k:
                                            state_dict[k.replace('module.', '')] = v
                                            del state_dict[k]
                                self.save_checkpoint(ckpt_filename, epoch, state_dict, optimizer)
                        else:
                            esnum += 1
                            if self.cfg.early_stop > 0 and esnum >= self.cfg.early_stop:
                                log = "Early Stopped at best epoch {}".format(epoch)
                                logs.append(log)
                                print(log)
                                # break
                                
                elif self.cfg.mode == 'finetune':
                    self.finetune(rank, net, train_loader, criterion, optimizer, epoch, self.cfg.epochs, logs)
                    if len(val_dataset) > 0:
                        self.validate(rank, net, val_loader, criterion, logs)
                
                if rank == 0 and (epoch+1) % self.cfg.save_freq == 0:
                    ckpt_dir = self.cfg.ckpt_dir
                    ckpt_filename = 'checkpoint_{:04d}.pth.tar'.format(epoch)
                    ckpt_filename = os.path.join(ckpt_dir, ckpt_filename)
                    state_dict = net.state_dict()
                    if world_size > 1:
                        for k, v in list(state_dict.items()):
                            if 'module.' in k:
                                state_dict[k.replace('module.', '')] = v
                                del state_dict[k]
                    self.save_checkpoint(ckpt_filename, epoch, state_dict, optimizer)
        
        if self.cfg.mode != 'pretrain':
            self.validate(rank, net, test_loader, criterion, logs)
    
    def split_per_domain(self, dataset):
        indices_per_domain = defaultdict(list)
        for i, d in enumerate(dataset):
            indices_per_domain[d[2].item()].append(i)
        return indices_per_domain
    
    def pretrain(self, rank, net, train_loader, criterion, optimizer, epoch, num_epochs, logs):
        net.train()
        
        for batch_idx, data in enumerate(train_loader):
            features = data[0].cuda()
            domains = data[2].cuda()
            
            if self.cfg.neg_per_domain:
                all_logits = []
                all_targets = []
                for dom in self.all_domains:
                    dom_idx = torch.nonzero(domains == dom).squeeze()
                    if dom_idx.dim() == 0: dom_idx = dom_idx.unsqueeze(0)
                    if torch.numel(dom_idx):
                        dom_features = features[dom_idx]
                        logits, targets = net(dom_features)
                        all_logits.append(logits)
                        all_targets.append(targets)
                logits = torch.cat(all_logits, dim=0)
                targets = torch.cat(all_targets, dim=0)
            else:
                logits, targets = net(features)
                
            loss = criterion(logits, targets)

            if rank == 0:
                if batch_idx % self.cfg.log_freq == 0:
                    acc1, acc5 = self.accuracy(logits, targets, topk=(1, 5))
                    log = f'Epoch [{epoch+1}/{num_epochs}]-({batch_idx}/{len(train_loader)}) '
                    log += f'Loss: {loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
                    logs.append(log)
                    print(log)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def validate_pretrain(self, rank, net, val_loader, criterion, logs):
        net.eval()
        
        total_targets = []
        total_logits = []
        total_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                features = data[0].cuda()
                domains = data[2].cuda()
                
                if self.cfg.neg_per_domain:
                    all_logits = []
                    all_targets = []
                    for dom in self.all_domains:
                        dom_idx = torch.nonzero(domains == dom).squeeze()
                        if dom_idx.dim() == 0: dom_idx = dom_idx.unsqueeze(0)
                        if torch.numel(dom_idx):
                            dom_features = features[dom_idx]
                            logits, targets = net(dom_features)
                            all_logits.append(logits)
                            all_targets.append(targets)
                    logits = torch.cat(all_logits, dim=0)
                    targets = torch.cat(all_targets, dim=0)
                else:
                    logits, targets = net(features)
                
                total_loss += criterion(logits, targets)
                total_targets.append(targets)
                total_logits.append(logits)
            
            if len(total_targets) > 0:
                total_targets = torch.cat(total_targets, dim=0)
                total_logits = torch.cat(total_logits, dim=0)
                acc1, acc5 = self.accuracy(total_logits, total_targets, topk=(1, 5))
                total_loss /= len(val_loader)
                
                log = f'Validation Loss: {total_loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
                self.write_log(rank, logs, log)
                
                return total_loss.item()
    
    def finetune(self, rank, net, train_loader, criterion, optimizer, epoch, num_epochs, logs):
        net.eval()
        
        for batch_idx, data in enumerate(train_loader):
            features = data[0].cuda()
            targets = data[1].cuda()
            
            logits = net(features)
            loss = criterion(logits, targets)
            
            if batch_idx % self.cfg.log_freq == 0:
                acc1, acc5 = self.accuracy(logits, targets, topk=(1, 5))
                log = f'Epoch [{epoch+1}/{num_epochs}]-({batch_idx}/{len(train_loader)}) '
                log += f'Loss: {loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
                self.write_log(rank, logs, log)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def validate(self, rank, net, val_loader, criterion, logs):
        net.eval()
        
        total_targets = []
        total_logits = []
        total_loss = 0
        for batch_idx, data in enumerate(val_loader):
            features = data[0].cuda()
            targets = data[1].cuda()
            
            logits = net(features)
            total_loss += criterion(logits, targets)
            total_targets.append(targets)
            total_logits.append(logits)
        
        total_targets = torch.cat(total_targets, dim=0)
        total_logits = torch.cat(total_logits, dim=0)
        acc1, acc5 = self.accuracy(total_logits, total_targets, topk=(1, 5))
        f1, recall, precision = self.scores(total_logits, total_targets)
        total_loss /= len(val_loader)
        
        log = f'Validation Loss: {total_loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
        log += f', F1: {f1.item():.2f}, Recall: {recall.item():.2f}, Precision: {precision.item():.2f}'
        self.write_log(rank, logs, log)
    
    def save_checkpoint(self, filename, epoch, state_dict, optimizer):
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        state = {
            'epoch': epoch+1,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, filename)
                
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
    