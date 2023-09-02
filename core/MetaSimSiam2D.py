import os
import random

from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.models as models
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from net.resnet import ResNet18, ResNet18_meta

class MetaPredictor(nn.Module):
    def __init__(self, dim, pred_dim=1):
        super(MetaPredictor, self).__init__()
        self.vars = nn.ParameterList()
        
        linear1 = nn.Linear(dim, pred_dim, bias=False)
        w = nn.Parameter(torch.ones_like(linear1.weight))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        
        bn1 = nn.BatchNorm1d(pred_dim)
        w = nn.Parameter(torch.ones_like(bn1.weight))
        b = nn.Parameter(torch.zeros_like(bn1.bias))
        running_mean = nn.Parameter(torch.zeros_like(bn1.running_mean), requires_grad=False)
        running_var = nn.Parameter(torch.ones_like(bn1.running_var), requires_grad=False)
        num_batches_tracked = nn.Parameter(torch.zeros_like(bn1.num_batches_tracked), requires_grad=False)
        self.vars.extend([w, b, running_mean, running_var, num_batches_tracked])
        
        linear2 = nn.Linear(pred_dim, dim, bias=False)
        w = nn.Parameter(torch.ones_like(linear2.weight))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
    
    def forward(self, x, vars=None, bn_training=True):
        if vars is None:
            vars = self.vars
            
        x = F.linear(x, weight=vars[0])
        x = F.batch_norm(x, vars[3], vars[4], vars[1], vars[2], training=bn_training)
        x = F.relu(x)
        x = F.linear(x, weight=vars[6])
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


class MetaSimSiamNet(nn.Module):
    def __init__(self, backbone='resnet18', out_dim=128, pred_dim=64, mlp=True):
        super(MetaSimSiamNet, self).__init__()
        self.vars = nn.ParameterList()
        
        self.out_dim = out_dim
        self.mlp = mlp
        self.pred_dim = pred_dim
        self.backbone = backbone
        
        if backbone == 'resnet18':
            encoder = ResNet18_meta(num_classes=out_dim, mlp=mlp)
            self.vars.extend(encoder.parameters())
            if mlp:
                self.vars[-1].requires_grad = False
                bn = nn.BatchNorm1d(out_dim, affine=False)
                running_mean = nn.Parameter(torch.zeros_like(bn.running_mean), requires_grad=False)
                running_var = nn.Parameter(torch.ones_like(bn.running_var), requires_grad=False)
                num_batches_tracked = nn.Parameter(torch.zeros_like(bn.num_batches_tracked), requires_grad=False)
                self.vars.extend([running_mean, running_var, num_batches_tracked])
        
        predictor = MetaPredictor(dim=out_dim, pred_dim=pred_dim)
        self.vars.extend(predictor.parameters())

    def forward(self, x1, x2, vars=None):
        if vars is None:
            vars = self.vars
        var_idx = 0
        
        if self.backbone == 'resnet18':
            encoder = ResNet18_meta(num_classes=self.out_dim, mlp=self.mlp)
            enc_vars = vars[:len(encoder.parameters())]
            var_idx += len(encoder.parameters())
            if self.mlp:
                enc_vars[-1] = None
                z1 = encoder(x1, vars=enc_vars, bn_training=True)
                z2 = encoder(x2, vars=enc_vars, bn_training=True)
                z1 = F.batch_norm(z1, vars[var_idx], vars[var_idx+1], training=True)
                z2 = F.batch_norm(z2, vars[var_idx], vars[var_idx+1], training=True)
                var_idx += 3
            else:
                z1 = encoder(x1, vars=enc_vars, bn_training=True)
                z2 = encoder(x2, vars=enc_vars, bn_training=True)
        
        predictor = MetaPredictor(dim=self.out_dim, pred_dim=self.pred_dim)
        pred_vars = vars[var_idx:]
        p1 = predictor(z1, vars=pred_vars, bn_training=True)
        p2 = predictor(z2, vars=pred_vars, bn_training=True)
        
        return p1, p2, z1.detach(), z2.detach()

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


class MetaSimSiamClassifier(nn.Module):
    def __init__(self, backbone, num_cls, mlp=True):
        super(MetaSimSiamClassifier, self).__init__()
        self.encoder = None
        if backbone == 'resnet18':
            self.encoder = ResNet18(num_classes=num_cls, mlp=mlp)
        
    def forward(self, x):
        x = self.encoder(x)
        return x


class MetaSimSiam2DLearner:
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
        net = MetaSimSiamNet(self.cfg.backbone, self.cfg.out_dim, self.cfg.pred_dim, self.cfg.pretrain_mlp)
        if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
            cls_net = MetaSimSiamClassifier(self.cfg.backbone, self.cfg.num_cls, self.cfg.finetune_mlp)
        
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
            
            if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
                cls_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(cls_net)
                cls_net.cuda()
                cls_net = nn.parallel.DistributedDataParallel(cls_net, device_ids=[rank], find_unused_parameters=True)
                
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
            
            if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
                cls_net.cuda()

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
                if self.cfg.no_vars:
                    for i, (k, v) in enumerate(state.items()):
                        new_k = list(net.state_dict().keys())[i]
                        new_state[new_k] = v
                else:
                    for k, v in list(state.items()):
                        if world_size > 1:
                            k = 'module.' + k
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
            
            if self.cfg.domain_adaptation:
                if rank == 0:
                    log = "Domain adaptation is enabled"
                    logs.append(log)
                    print(log)
                if world_size > 1:
                    train_sampler.set_epoch(0)
                net.train()
                net.zero_grad()
                
                s_x1 = torch.stack([x[0][0] for x in meta_train_dataset]).cuda()
                s_x2 = torch.stack([x[0][1] for x in meta_train_dataset]).cuda()
                enc_params = self.meta_train(rank, net, s_x1, s_x2, criterion, self.cfg.log_steps, logs)
                # TODO: self.meta_eval()
            else:
                enc_params = list(net.parameters())
                
            if rank == 0:
                log = "Loading encoder parameters to classifier"
                logs.append(log)
                print(log)
            
            enc_dict = {}
            for i, (k, v) in enumerate(cls_net.state_dict().items()):
                if not 'fc' in k:
                    enc_dict[k] = enc_params[i]
            msg = cls_net.load_state_dict(enc_dict, strict=False)
            if rank == 0:
                log = "Missing keys: {}".format(msg.missing_keys)
                logs.append(log)
                print(log)
            net = cls_net
            
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
        # if self.cfg.mode == 'finetune': scheduler = StepLR(optimizer, step_size=self.cfg.lr_decay_step, gamma=self.cfg.lr_decay)
        
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
            
            for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
                if world_size > 1:
                    train_sampler.set_epoch(epoch)
                    if len(val_dataset) > 0:
                        val_sampler.set_epoch(epoch)
                    test_sampler.set_epoch(epoch)
                
                if self.cfg.mode == 'pretrain':
                    # d_s_x1s, d_s_x2s, d_q_x1s, d_q_x2s = self.gen_per_domain_tasks(
                    #     train_dataset, sampled_indices_by_domain, self.cfg.task_size, self.cfg.num_task)
                    # r_s_x1s, r_s_x2s, r_q_x1s, r_q_x2s = self.gen_random_tasks(
                    #     meta_train_dataset, self.cfg.task_size, self.cfg.multi_cond_num_task)
                    # s_x1s = d_s_x1s + r_s_x1s
                    # s_x2s = d_s_x2s + r_s_x2s
                    # q_x1s = d_q_x1s + r_q_x1s
                    # q_x2s = d_q_x2s + r_q_x2s
                    supports, queries = self.gen_per_domain_tasks(
                        train_dataset, sampled_indices_by_domain, self.cfg.task_size, self.cfg.num_task//world_size)
                    rand_supports, rand_queries = self.gen_random_tasks(
                        meta_train_dataset, self.cfg.task_size, self.cfg.multi_cond_num_task//world_size)
                    supports = supports + rand_supports
                    queries = queries + rand_queries
                    # self.pretrain(rank, net, s_x1s, s_x2s, q_x1s, q_x2s, criterion, optimizer, epoch, self.cfg.epochs, logs)
                    self.pretrain(rank, net, supports, queries, criterion, optimizer, epoch, self.cfg.epochs, logs)
                    
                elif self.cfg.mode == 'finetune':
                    self.finetune(rank, net, train_loader, criterion, optimizer, epoch, self.cfg.epochs, logs)
                    # if len(val_dataset) > 0: self.validate_finetune(rank, net, val_loader, criterion, logs)
                
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
        # s_x1s = []
        # s_x2s = []
        # q_x1s = []
        # q_x2s = []
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
                    
                # s_x1 = torch.stack([x[0][0] for x in support])
                # s_x2=  torch.stack([x[0][1] for x in support])
                # q_x1 = torch.stack([x[0][0] for x in query])
                # q_x2 = torch.stack([x[0][1] for x in query])
                
                # s_x1s.append(s_x1)
                # s_x2s.append(s_x2)
                # q_x1s.append(q_x1)
                # q_x2s.append(q_x2)
                supports.append(support)
                queries.append(query)
        
        # return s_x1s, s_x2s, q_x1s, q_x2s
        return supports, queries
    
    def gen_random_tasks(self, dataset, task_size, num_task):
        # s_x1s = []
        # s_x2s = []
        # q_x1s = []
        # q_x2s = []
        supports = []
        queries = []
        with torch.no_grad():
            for _ in range(num_task):
                indices = list(range(len(dataset)))
                random.shuffle(indices)
                
                support = Subset(dataset, indices[:task_size])
                query = Subset(dataset, indices[task_size:2*task_size])
                
                # s_x1 = torch.stack([x[0][0] for x in support])
                # s_x2=  torch.stack([x[0][1] for x in support])
                # q_x1 = torch.stack([x[0][0] for x in query])
                # q_x2 = torch.stack([x[0][1] for x in query])
                    
                # s_x1s.append(s_x1)
                # s_x2s.append(s_x2)
                # q_x1s.append(q_x1)
                # q_x2s.append(q_x2)
                supports.append(support)
                queries.append(query)
        
        # return s_x1s, s_x2s, q_x1s, q_x2s
        return supports, queries
    
    def pretrain(self, rank, net, supports, queries, criterion, optimizer, epoch, num_epochs, logs):
        net.train()
        net.zero_grad()
        
        def get_features(batch):
            x1 = []
            x2 = []
            for x in batch:
                x1.append(x[0][0])
                x2.append(x[0][1])
            return torch.stack(x1).cuda(), torch.stack(x2).cuda()
        
        losses = []
        for idx in range(len(supports)):
            support_loader = DataLoader(supports[idx], batch_size=len(supports[idx]), collate_fn=get_features)
            s_x1, s_x2 = next(iter(support_loader))
            fast_weights = self.meta_train(rank, net, s_x1, s_x2, criterion, self.cfg.log_steps, logs)
            
            query_loader = DataLoader(queries[idx], batch_size=len(queries[idx]), collate_fn=get_features)
            q_x1, q_x2 = next(iter(query_loader))
            p1, p2, z1, z2 = net(q_x1, q_x2, vars=fast_weights)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            losses.append(loss)
            
            if idx % self.cfg.log_freq == 0 and rank == 0:
                log = f'Epoch [{epoch+1}/{num_epochs}]-({idx}/{len(supports)}) '
                log += f'Loss: {loss.item():.4f}'
                logs.append(log)
                print(log)
        
        losses = torch.stack(losses, dim=0)
        loss = torch.sum(losses)/len(supports)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def meta_train(self, rank, net, x1, x2, criterion, log_steps=False, logs=None):
        fast_weights = list(net.parameters())
        for step in range(self.cfg.task_steps):
            p1, p2, z1, z2 = net(x1, x2, vars=fast_weights)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            grad = torch.autograd.grad(loss, [p for p in fast_weights if p.requires_grad])
            
            new_weights = []
            grad_idx = 0
            for p in fast_weights:
                if p.requires_grad:
                    new_weights.append(p - self.cfg.task_lr * grad[grad_idx])
                    grad_idx += 1
                else:
                    new_weights.append(p)
            fast_weights = new_weights
            
            if log_steps and rank == 0:
                log = f'\tStep [{step}/{self.cfg.task_steps}] Loss: {loss.item():.4f}'
                logs.append(log)
                print(log)
        
        return fast_weights
            
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
