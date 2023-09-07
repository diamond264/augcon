import os
import time
import numpy as np
import sklearn.metrics as metrics
import clip

from sklearn.neighbors import KNeighborsClassifier as KNN

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.models as models
import torch.multiprocessing as mp

# from datautils.SimCLR_dataset import subject_collate
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from net.resnet import ResNet18, ModifiedResNet50, ResNet18_meta
from net.convnetDigit5 import CNN

class Predictor(nn.Module):
    def __init__(self, dim, pred_dim=1):
        super(Predictor, self).__init__()
        self.layer = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                   nn.BatchNorm1d(pred_dim),
                                   nn.ReLU(inplace=True), # hidden layer
                                   nn.Linear(pred_dim, dim))
    
    def forward(self, x):
        x = self.layer(x)
        return x


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.adapter(x)
        return x


class SimSiamNet(nn.Module):
    def __init__(self, backbone='resnet18', out_dim=128, pred_dim=64, mlp=True, adapter=False):
        super(SimSiamNet, self).__init__()
        self.encoder = None
        if backbone == 'resnet18':
            self.encoder = ResNet18(num_classes=out_dim, mlp=mlp)
        elif backbone == 'resnet50':
            self.encoder = ModifiedResNet50(num_classes=out_dim, mlp=mlp, adapter=adapter)
        elif backbone == 'cnn':
            self.encoder = CNN(num_classes=out_dim, mlp=mlp)

        if mlp:
            self.encoder.fc[6].bias.requires_grad = False
            self.encoder.fc = nn.Sequential(self.encoder.fc, nn.BatchNorm1d(out_dim, affine=False))
        
        self.predictor = Predictor(dim=out_dim, pred_dim=pred_dim)

    def forward(self, feature, aug_feature):
        z = self.encoder(feature)
        p = self.predictor(z)
        aug_z = self.encoder(aug_feature)
        aug_p = self.predictor(aug_z)

        return p, aug_p, z.detach(), aug_z.detach()


class SimSiamClassifier(nn.Module):
    def __init__(self, backbone, num_cls, mlp=True, adapter=False):
        super(SimSiamClassifier, self).__init__()
        self.encoder = None
        if backbone == 'resnet18':
            self.encoder = ResNet18(num_classes=num_cls, mlp=mlp)
        elif backbone == 'resnet50':
            self.encoder = ModifiedResNet50(num_classes=num_cls, mlp=mlp, adapter=adapter)
        elif backbone == 'cnn':
            self.encoder = CNN(num_classes=num_cls, mlp=mlp)
        
    def forward(self, x):
        x = self.encoder(x)
        return x


class SimSiam2DLearner:
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
        self.logger.info("Executing SimSiam(2D)")
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
        if self.cfg.mode == 'pretrain' or self.cfg.mode == 'eval_pretrain':
            net = SimSiamNet(self.cfg.backbone, self.cfg.out_dim, self.cfg.pred_dim, self.cfg.pretrain_mlp, self.cfg.adapter)
        elif self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
            net = SimSiamClassifier(self.cfg.backbone, self.cfg.n_way, self.cfg.finetune_mlp, self.cfg.adapter)
        
        # Setting Distributed Data Parallel configuration
        if world_size > 1:
            dist.init_process_group(backend='nccl', init_method=self.cfg.dist_url,
                                    world_size=world_size, rank=rank)
            torch.cuda.set_device(rank)
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net.cuda()
            net = nn.parallel.DistributedDataParallel(net, device_ids=[rank], find_unused_parameters=True)
            
            train_sampler = DistributedSampler(train_dataset)
            test_sampler = DistributedSampler(test_dataset)
            if len(val_dataset) > 0:
                val_sampler = DistributedSampler(val_dataset)
            
            train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size//world_size,
                                      shuffle=False, sampler=train_sampler,
                                      num_workers=self.cfg.num_workers, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size//world_size,
                                     shuffle=False, sampler=test_sampler,
                                     num_workers=self.cfg.num_workers, drop_last=True)
            if len(val_dataset) > 0:
                val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size//world_size,
                                        shuffle=False, sampler=val_sampler,
                                        num_workers=self.cfg.num_workers, drop_last=True)
            self.write_log(rank, logs, "DDP is used for training - training {} instances for each worker".format(len(list(train_sampler))))
        else:
            torch.cuda.set_device(rank)
            net.cuda()
            
            train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size,
                                      shuffle=True, num_workers=self.cfg.num_workers, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size,
                                     shuffle=True, num_workers=self.cfg.num_workers, drop_last=True)
            if len(val_dataset) > 0:
                val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size,
                                        shuffle=True, num_workers=self.cfg.num_workers, drop_last=True)
            self.write_log(rank, logs, "Single GPU is used for training - training {} instances for each worker".format(len(train_dataset)))
        
        # Define criterion
        if self.cfg.mode == 'pretrain' or self.cfg.mode == 'eval_pretrain':
            criterion = nn.CosineSimilarity(dim=1).cuda()
        elif self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
            criterion = nn.CrossEntropyLoss().cuda()
        
        # For pretraining, load pretrained model
        if self.cfg.mode == 'pretrain':
            if self.cfg.pretrained == 'clip':
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if self.cfg.backbone == 'resnet50':
                    self.write_log(rank, logs, "Loading pretrained model from CLIP - RN50")
                    clip_model, _ = clip.load("RN50", device=device)
                    clip_state = clip_model.visual.state_dict()
                    
                    new_state = {}
                    for k, v in clip_state.items():
                        k = 'encoder.' + k
                        if world_size > 1:
                            k = 'module.' + k
                        if not k in net.state_dict().keys():
                            self.write_log(rank, logs, f'Unrecognized key in CLIP: {k}')
                        if not 'fc' in k:
                            new_state[k] = v
                        
                    msg = net.load_state_dict(new_state, strict=False)
                    self.write_log(rank, logs, f'Missing keys: {msg.missing_keys}')
                    del clip_state, clip_model
                
                if self.cfg.adapter:
                    for name, param in net.named_parameters():
                        if 'adapter' in name or 'fc' in name or 'predictor' in name: param.requires_grad = True
                        else: param.requires_grad = False
        
        # For finetuning, load pretrained model
        if self.cfg.mode == 'finetune':
            if self.cfg.pretrained == 'clip':
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if self.cfg.backbone == 'resnet50':
                    self.write_log(rank, logs, "Loading pretrained model from CLIP - RN50")
                    clip_model, _ = clip.load("RN50", device=device)
                    clip_state = clip_model.visual.state_dict()
                    
                    new_state = {}
                    for k, v in clip_state.items():
                        k = 'encoder.' + k
                        if world_size > 1:
                            k = 'module.' + k
                        if not k in net.state_dict().keys():
                            self.write_log(rank, logs, f'Unrecognized key in CLIP: {k}')
                        if not 'fc' in k:
                            new_state[k] = v
                        
                    msg = net.load_state_dict(new_state, strict=False)
                    self.write_log(rank, logs, f'Missing keys: {msg.missing_keys}')
                    del clip_state, clip_model
            elif os.path.isfile(self.cfg.pretrained):
                loc = 'cuda:{}'.format(rank)
                state = torch.load(self.cfg.pretrained, map_location=loc)['state_dict']
                self.write_log(rank, logs, "Loading pretrained model from checkpoint - {}".format(self.cfg.pretrained))

                # To handle the case where the model is trained in multi-gpu environment
                new_state = {}
                for k, v in list(state.items()):
                    if world_size > 1: k = 'module.' + k
                    if k in net.state_dict().keys() and not 'fc' in k: new_state[k] = v
                msg = net.load_state_dict(new_state, strict=False)
                self.write_log(rank, logs, "Missing keys: {}".format(msg.missing_keys))
            else:
                self.write_log(rank, logs, "No checkpoint found at '{}'".format(self.cfg.pretrained))
            
            # Freezing the encoder
            if self.cfg.freeze:
                self.write_log(rank, logs, "Freezing the encoder")
                for name, param in net.named_parameters():
                    if not 'fc' in name: param.requires_grad = False
                    else: param.requires_grad = True
        
        # Define optimizer
        parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
        if self.cfg.optimizer == 'sgd':
            optimizer = torch.optim.SGD(parameters, self.cfg.lr,
                                        momentum=self.cfg.momentum,
                                        weight_decay=self.cfg.wd)
        elif self.cfg.optimizer == 'adam':
            optimizer = torch.optim.Adam(parameters, self.cfg.lr,
                                         weight_decay=self.cfg.wd)
        # if self.cfg.mode == 'finetune':
        #     scheduler = StepLR(optimizer, step_size=self.cfg.lr_decay_step, gamma=self.cfg.lr_decay)
        
        # Load checkpoint if exists
        if os.path.isfile(self.cfg.resume):
            self.write_log(rank, logs, "Loading state_dict from checkpoint - {}".format(self.cfg.resume))
            loc = 'cuda:{}'.format(rank)
            state = torch.load(self.cfg.resume, map_location=loc)
            
            for k, v in list(state['state_dict'].items()):
                new_k = k
                if world_size > 1:
                    new_k = 'module.' + k
                if new_k in net.state_dict().keys():
                    state['state_dict'][new_k] = v
                if k not in net.state_dict().keys():
                    state['state_dict'].pop(k,None)
            
            net.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            self.cfg.start_epoch = state['epoch']
        
        # Performing the main task
        if self.cfg.mode == 'eval_pretrain':
            self.validate_pretrain(rank, net, test_loader, criterion, logs)
        elif self.cfg.mode == 'eval_finetune':
            self.validate_finetune(rank, net, test_loader, criterion, logs)
        else:
            # loss_best = 0
            for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
                if world_size > 1:
                    train_sampler.set_epoch(epoch)
                    test_sampler.set_epoch(epoch)
                    if len(val_dataset) > 0:
                        val_sampler.set_epoch(epoch)
                
                if self.cfg.mode == 'pretrain':
                    self.pretrain(rank, net, train_loader, criterion,
                                  optimizer, epoch, self.cfg.epochs, logs)
                    
                elif self.cfg.mode == 'finetune':
                    self.finetune(rank, net, train_loader, criterion,
                                  optimizer, epoch, self.cfg.epochs, logs)
                    if (epoch+1) % 10 == 0:
                        self.validate_finetune(rank, net, test_loader, criterion, logs)
                    # if len(val_dataset) > 0:
                    #     self.validate_finetune(rank, net, val_loader, criterion, logs)
                
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
            
            if self.cfg.mode == 'finetune':
                self.validate_finetune(rank, net, test_loader, criterion, logs)
    
    def pretrain(self, rank, net, train_loader, criterion, optimizer, epoch, num_epochs, logs):
        if self.cfg.pretrained == 'clip':
            net.eval()
            net.module.encoder.adapter.train()
            net.module.encoder.fc.train()
        else:
            net.train()
        
        for batch_idx, data in enumerate(train_loader):
            features = data[0][0].cuda()
            pos_features = data[0][1].cuda()
            
            p1, p2, z1, z2 = net(features, pos_features)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            
            std = torch.std(torch.cat((z1, z2), dim=0))
            if batch_idx % self.cfg.log_freq == 0:
                log = f'Epoch [{epoch+1}/{num_epochs}]-({batch_idx}/{len(train_loader)}) '
                log += f'Loss: {loss.item():.4f}\tStd: {std.item():.4f}'
                self.write_log(rank, logs, log)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    def validate_pretrain(self, rank, net, val_loader, criterion, logs):
        net.eval()
        
        with torch.no_grad():
            stds = []
            losses = []
            KNN_accs = []
            for batch_idx, data in enumerate(val_loader):
                features = data[0][0].cuda()
                pos_features = data[0][1].cuda()
                
                p1, p2, z1, z2 = net(features, pos_features)
                stds.append(torch.std(torch.cat((z1, z2), dim=0)).item())
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                losses.append(loss.item())
                
                length = z1.shape[0]
                KNN_classifier = KNN(n_neighbors=5)
                KNN_classifier.fit(z1.detach().cpu().numpy()[:length], y=data[1].numpy()[:length])
                KNN_pred = KNN_classifier.predict(z1.detach().cpu().numpy()[length:])
                KNN_acc = np.mean(KNN_pred == data[1].numpy()[length:])*100
            
            std = np.mean(stds)
            loss = np.mean(losses)
            KNN_acc = np.mean(KNN_accs)
            self.write_log(rank, logs, f'[Pretrain] Validation Loss: {loss:.4f}\tStd: {std:.4f}\tKNN Acc: {KNN_acc:.2f}')
            
            return loss
    
    def finetune(self, rank, net, train_loader, criterion, optimizer, epoch, num_epochs, logs):
        net.eval()
        
        for batch_idx, data in enumerate(train_loader):
            features = data[0].cuda()
            targets = data[1].cuda()
            
            logits = net(features)
            loss = criterion(logits, targets)
            
            if rank == 0:
                acc1, acc5 = self.accuracy(logits, targets, topk=(1, 5))
                log = f'Epoch [{epoch+1}/{num_epochs}]-({batch_idx}/{len(train_loader)}) '
                log += f'\tLoss: {loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
                self.write_log(rank, logs, log)
            
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
                features = data[0].cuda()
                targets = data[1].cuda()
                
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
            
            log = f'[Finetune] Validation Loss: {total_loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
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
