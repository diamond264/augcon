import os
import random

from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.models as models
import torch.multiprocessing as mp

from torch.autograd import Variable
from torch.utils.data import DataLoader, DistributedSampler, Subset

from net.resnet import ResNet18, ResNet18_meta

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


class SimSiamNet(nn.Module):
    def __init__(self, backbone='resnet18', out_dim=128, pred_dim=64, mlp=True):
        super(SimSiamNet, self).__init__()
        self.encoder = None
        if backbone == 'resnet18':
            self.encoder = ResNet18(num_classes=out_dim, mlp=mlp)
            if mlp:
                self.encoder.fc[6].bias.requires_grad = False
                self.encoder.fc = nn.Sequential(self.encoder.fc, nn.BatchNorm1d(out_dim, affine=False))
        
        self.predictor = Predictor(dim=out_dim, pred_dim=pred_dim)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        p1 = self.predictor(z1)
        z2 = self.encoder(x2)
        p2 = self.predictor(z2)
        
        return p1, p2, z1.detach(), z2.detach()


class SimSiamClassifier(nn.Module):
    def __init__(self, backbone, num_cls, mlp=True):
        super(SimSiamClassifier, self).__init__()
        self.encoder = None
        if backbone == 'resnet18':
            self.encoder = ResNet18(num_classes=num_cls, mlp=mlp)
        
    def forward(self, x):
        x = self.encoder(x)
        return x


class ReptileSimSiamLearner:
    def __init__(self, cfg, gpu, logger):
        self.cfg = cfg
        self.gpu = gpu
        self.logger = logger
        
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
            net = SimSiamNet(self.cfg.backbone, self.cfg.out_dim, self.cfg.pred_dim, self.cfg.mlp)
        elif self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
            net = SimSiamClassifier(self.cfg.backbone, self.cfg.num_cls, self.cfg.mlp)
        
        # DDP setting
        if world_size > 1:
            dist.init_process_group(backend='nccl',
                                    init_method=self.cfg.dist_url,
                                    world_size=world_size,
                                    rank=rank)
            torch.cuda.set_device(rank)
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net.cuda()
            net = nn.parallel.DistributedDataParallel(net, device_ids=[rank], find_unused_parameters=True)
            
            train_sampler = DistributedSampler(train_dataset)
            if len(val_dataset) > 0:
                val_sampler = DistributedSampler(val_dataset)
            test_sampler = DistributedSampler(test_dataset)
            meta_train_dataset = Subset(train_dataset, list(train_sampler))
            
            # collate_fn = subject_collate if self.cfg.mode == 'pretrain' else None
            collate_fn = None
            train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size//world_size,
                                      shuffle=False, sampler=train_sampler, collate_fn=collate_fn,
                                      num_workers=self.cfg.num_workers, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size//world_size,
                                     shuffle=False, sampler=test_sampler, collate_fn=collate_fn,
                                     num_workers=self.cfg.num_workers, drop_last=True)
            if len(val_dataset) > 0:
                val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size//world_size,
                                        shuffle=False, sampler=val_sampler, collate_fn=collate_fn,
                                        num_workers=self.cfg.num_workers, drop_last=True)
            if rank == 0:
                log = "DDP is used for training - training {} instances for each worker".format(len(list(train_sampler)))
                logs.append(log)
                print(log)
        else:
            torch.cuda.set_device(rank)
            net.cuda()
            
            meta_train_dataset = train_dataset
            
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
            if rank == 0:
                log = "Single GPU is used for training - training {} instances for each worker".format(len(train_dataset))
                logs.append(log)
                print(log)
        
        # Define criterion
        if self.cfg.mode == 'pretrain' or self.cfg.mode == 'eval_pretrain':
            criterion = nn.CosineSimilarity(dim=1).cuda()
        elif self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
            criterion = nn.CrossEntropyLoss().cuda()
        
        # For finetuning, load pretrained model
        if self.cfg.mode == 'finetune':
            if os.path.isfile(self.cfg.pretrained):
                if rank == 0:
                    log = "Loading pretrained model from checkpoint - {}".format(self.cfg.pretrained)
                    logs.append(log)
                    print(log)
                loc = 'cuda:{}'.format(rank)
                state = torch.load(self.cfg.pretrained, map_location=loc)['state_dict']
                
                # if rank == 0:
                #     print(state.keys())
                #     print(net.state_dict().keys())
                #     print(len(state.keys()))
                #     print(len(net.state_dict().keys()))
                # assert(0)
                
                new_state = {}
                for k, v in list(state.items()):
                    if world_size > 1:
                        k = 'module.' + k
                    if k in net.state_dict().keys() and not 'fc' in k:
                        new_state[k] = v
                
                msg = net.load_state_dict(new_state, strict=False)
                if rank == 0:
                    log = "Missing keys: {}".format(msg.missing_keys)
                    logs.append(log)
                    print(log)
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
                    if not 'fc' in name:
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
        
        # if self.cfg.mode == 'finetune':
        #     scheduler = StepLR(optimizer, step_size=self.cfg.lr_decay_step, gamma=self.cfg.lr_decay)
        
        # Load checkpoint if exists
        if os.path.isfile(self.cfg.resume):
            if rank == 0:
                log = "Loading state_dict from checkpoint - {}".format(self.cfg.resume)
                logs.append(log)
                print(log)
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
        
        # Handling the modes (train or eval)
        if self.cfg.mode == 'eval_pretrain':
            self.validate_pretrain(rank, net, test_loader, criterion, logs)
        elif self.cfg.mode == 'eval_finetune':
            self.validate_finetune(rank, net, test_loader, criterion, logs)
        else:
            if self.cfg.mode == 'pretrain':
                if rank == 0:
                    log = "Getting indices by domain..."
                    logs.append(log)
                    print(log)
                indices_by_domain = train_dataset.get_indices_by_domain()
                sampled_indices_by_domain = defaultdict(list)
                for k, v in indices_by_domain.items():
                    indices = []
                    for idx in v:
                        if idx in meta_train_dataset.indices:
                            indices.append(idx)
                    sampled_indices_by_domain[k] = indices
                
            if rank == 0:
                log = "Perform training"
                logs.append(log)
                print(log)
            
            optimizer_state = None
            for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
                if world_size > 1:
                    train_sampler.set_epoch(epoch)
                    if len(val_dataset) > 0:
                        val_sampler.set_epoch(epoch)
                    test_sampler.set_epoch(epoch)
                
                if self.cfg.mode == 'pretrain':
                    supports, queries = self.gen_per_domain_tasks(
                        train_dataset, sampled_indices_by_domain, self.cfg.task_size, self.cfg.num_task//world_size)
                    rand_supports, rand_queries = self.gen_random_tasks(
                        meta_train_dataset, self.cfg.task_size, self.cfg.multi_cond_num_task//world_size)
                    supports = supports + rand_supports
                    queries = queries + rand_queries
                    optimizer_state = self.pretrain(rank, net, supports, criterion, epoch, self.cfg.epochs, logs)
                    
                elif self.cfg.mode == 'finetune':
                    self.finetune(rank, net, train_loader, criterion, optimizer, epoch, self.cfg.epochs, logs)
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
    
    def gen_per_domain_tasks(self, dataset, indices_by_domain, task_size, num_task):
        domains = list(indices_by_domain.keys())
        supports = []
        queries = []
        with torch.no_grad():
            for i in range(num_task):
                domain = random.choice(domains)
                sample_indices = indices_by_domain[domain]
                random.shuffle(sample_indices)
                
                support = Subset(dataset, sample_indices[:task_size])
                if len(support) < task_size:
                    support = support*(task_size//len(support))+support[:task_size%len(support)]
                query = Subset(dataset, sample_indices[task_size:2*task_size])
                if len(query) < task_size:
                    query = query*(task_size//len(query))+query[:task_size%len(query)]
                
                supports.append(support)
                queries.append(query)
        
        return supports, queries
    
    def gen_random_tasks(self, dataset, task_size, num_task):
        supports = []
        queries = []
        with torch.no_grad():
            for _ in range(num_task):
                indices = list(range(len(dataset)))
                random.shuffle(indices)
                
                support = Subset(dataset, indices[:task_size])
                query = Subset(dataset, indices[task_size:2*task_size])
                supports.append(support)
                queries.append(query)
        
        return supports, queries
    
    def get_optimizer(self, net, state=None):
        optimizer = torch.optim.Adam(net.parameters(), lr=self.cfg.lr, betas=(0, 0.999))
        if state is not None:
            optimizer.load_state_dict(state)
        return optimizer
    
    def pretrain(self, rank, net, supports, criterion, epoch, num_epochs, logs):
        net.train()
        num_tasks = len(supports)
        
        def get_features(batch):
            x1 = []
            x2 = []
            for x in batch:
                x1.append(x[0][0])
                x2.append(x[0][1])
            return torch.stack(x1).cuda(), torch.stack(x2).cuda()
        
        old_weights = deepcopy(net.state_dict())
        weight_updates = {name: 0 for name in old_weights}
        losses = []
        for task_idx in range(num_tasks):
            support = supports[task_idx]
            batch_indices = [torch.randint(len(support), size=(self.cfg.inner_batch_size,)) for _ in range(self.cfg.inner_steps)]
            support_loader = DataLoader(support, batch_sampler=batch_indices, collate_fn=get_features)
            net.zero_grad()
            optimizer = self.get_optimizer(net)
            for inner_step, (s_x1, s_x2) in enumerate(support_loader):
                p1, p2, z1, z2 = net(s_x1, s_x2)
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if self.cfg.log_steps and rank == 0:
                    log = '\t'
                    if inner_step == self.cfg.inner_steps-1: log += '(Final) '
                    log += f'Task({task_idx}) Step [{inner_step}/{self.cfg.inner_steps}] Loss: {loss.item():.4f}'
                    logs.append(log)
                    print(log)
            
            weight_updates = {name: weight_updates[name]+net.state_dict()[name]-old_weights[name] 
                              for name in old_weights}
            net.load_state_dict(old_weights)
            losses.append(loss)
        
        net.load_state_dict({name :
            old_weights[name]+weight_updates[name]*self.cfg.epsilone/num_tasks
            for name in old_weights})

        if task_idx % self.cfg.log_freq == 0 and rank == 0:
            log = f'Epoch [{epoch+1}/{num_epochs}] '
            log += f'Loss: {sum(losses)/len(losses):.4f}'
            logs.append(log)
            print(log)
            
    def validate_pretrain(self, rank, net, val_loader, criterion, logs):
        net.eval()
        
        total_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                features = data[1].cuda()
                pos_features = data[2].cuda()
                
                p1, p2, z1, z2 = net(features, pos_features)
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                total_loss += loss
                
            total_loss /= len(val_loader)
            
            if rank == 0:
                log = f'[Pretrain] Validation Loss: {total_loss.item():.4f}'
                logs.append(log)
                print(log)
            
            return total_loss.item()
    
    def finetune(self, rank, net, train_loader, criterion, optimizer, epoch, num_epochs, logs):
        if self.cfg.freeze: net.eval()
        else: net.train()
        
        for batch_idx, data in enumerate(train_loader):
            features = data[0].cuda()
            targets = data[1].cuda()
            
            logits = net(features)
            loss = criterion(logits, targets)
            
            if rank == 0:
                if batch_idx % self.cfg.log_freq == 0:
                    acc1, acc5 = self.accuracy(logits, targets, topk=(1, 5))
                    log = f'Epoch [{epoch+1}/{num_epochs}]-({batch_idx}/{len(train_loader)}) '
                    log += f'\tLoss: {loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
                    logs.append(log)
                    print(log)
            
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
                # f1, recall, precision = self.scores(total_logits, total_targets)
            total_loss /= len(val_loader)
            
            if rank == 0:
                log = f'[Finetune] Validation Loss: {total_loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
                # log += f', F1: {f1.item():.2f}, Recall: {recall.item():.2f}, Precision: {precision.item():.2f}'
                logs.append(log)
                print(log)
    
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
            batch_size = target.size(0)

            _, pred = output.max(1)
            correct = pred.eq(target)

            # Compute f1-score, recall, and precision for top-1 prediction
            tp = torch.logical_and(correct, target).sum()
            fp = pred.sum() - tp
            fn = target.sum() - tp
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1 = (2 * precision * recall) / (precision + recall + 1e-12)

            return f1, recall, precision
