import os
import random
import numpy as np

# import clip
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier as KNN

from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.models as models
import torch.multiprocessing as mp

from torchvision import transforms

from torch.autograd import Variable
from torch.utils.data import DataLoader, DistributedSampler, Subset

from net.resnet import ResNet18, ResNet50_meta_adapt, ResNet50
from net.convnetDigit5 import CNN

from data_loader.DomainNetDataset import DomainNetDataset
from RandAugment import RandAugment

from tqdm import tqdm

class SimSiamNet(nn.Module):
    def __init__(self, backbone='resnet18', out_dim=128, pred_dim=64, mlp=True, use_adapter=False, adapter_ratio=0, adapt_algorithm='simsiam', supervised_adaptation=False):
        super(SimSiamNet, self).__init__()
        self.encoder = None
        if backbone == 'resnet18':
            self.encoder = ResNet18(num_classes=out_dim, mlp=mlp)
        elif backbone == 'resnet50':
            self.encoder = ResNet50_meta_adapt(num_classes=out_dim, mlp=mlp, use_adapter=use_adapter, adapter_ratio=adapter_ratio)
        elif backbone == 'cnn':
            self.encoder = CNN(num_classes=out_dim, mlp=mlp)
        elif backbone == 'imagenet_resnet50':
            self.encoder = models.resnet50(num_classes=out_dim, zero_init_residual=True)
            if mlp:
                prev_dim = self.encoder.fc.weight.shape[1]
                self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                                nn.BatchNorm1d(prev_dim),
                                                nn.ReLU(inplace=True), # first layer
                                                nn.Linear(prev_dim, prev_dim, bias=False),
                                                nn.BatchNorm1d(prev_dim),
                                                nn.ReLU(inplace=True), # second layer
                                                self.encoder.fc,
                                                nn.BatchNorm1d(out_dim, affine=False)) # output layer
            self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN
        if mlp and backbone != 'imagenet_resnet50':
            self.encoder.fc[6].bias.requires_grad = False
            self.encoder.fc.append(nn.BatchNorm1d(out_dim, affine=False))
        
        self.predictor = nn.Sequential(nn.Linear(out_dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, out_dim)) # output layer
        self.adapt_algorithm = adapt_algorithm
        self.supervised_adaptation = supervised_adaptation

    # def forward(self, x1, x2, vars=None):
    #     if vars == None:
    #         vars = self.encoder.vars
    #     z1 = self.encoder(x1, vars)
    #     p1 = self.predictor(z1)
    #     z2 = self.encoder(x2, vars)
    #     p2 = self.predictor(z2)
        
    #     return p1, p2, z1.detach(), z2.detach()
    
    def forward(self, x1, x2, y=None, vars=None):
        if vars == None:
            vars = self.encoder.vars
        # compute query features
        z1 = self.encoder(x1, vars)  # queries: NxC
        z1 = nn.functional.normalize(z1, dim=1) 
        z2 = self.encoder(x2, vars)
        z2 = nn.functional.normalize(z2, dim=1)
        
        LARGE_NUM = 1e9
        batch_size = z1.size(0)

        labels = torch.arange(batch_size).cuda()
        masks = F.one_hot(torch.arange(batch_size), batch_size).cuda()
        # mask = ~torch.eye(batch_size, dtype=bool).cuda()

        logits_aa = torch.matmul(z1, z1.t())
        # logits_aa = logits_aa[mask].reshape(batch_size, batch_size-1)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(z2, z2.t())
        # logits_bb = logits_bb[mask].reshape(batch_size, batch_size-1)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(z1, z2.t())

        logits = torch.cat([torch.einsum('ij,ij->i', z1, z2).reshape(batch_size, 1), logits_aa], dim=1)#, logits_bb], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long).cuda()
        if not y is None:
            logits = logits_ab
            labels = torch.arange(batch_size).cuda()
            for i in range(len(logits)):
                for j in range(len(logits)):
                    if i != j and y[i] == y[j]:
                        logits[i][j] = -LARGE_NUM
        logits /= 0.1
        # logits = F.pad(logits, (0, full_batch_size*3-logits.shape[1]), "constant", -LARGE_NUM)

        return logits, labels
    
    # def zero_grad(self, vars=None):
    #     with torch.no_grad():
    #         if vars is None:
    #             for p in self.encoder.vars:
    #                 if p.grad is not None:
    #                     p.grad.zero_()
    #         else:
    #             for p in vars:
    #                 if p.grad is not None:
    #                     p.grad.zero_()
    
    # def parameters(self):
    #     return self.encoder.vars


class SimSiamClassifier(nn.Module):
    def __init__(self, backbone, num_cls, mlp=True, use_adapter=False, adapter_ratio=0):
        super(SimSiamClassifier, self).__init__()
        self.encoder = None
        if backbone == 'resnet18':
            self.encoder = ResNet18(num_classes=num_cls, mlp=mlp)
        elif backbone == 'resnet50':
            self.encoder = ResNet50_meta_adapt(num_classes=num_cls, mlp=mlp, use_adapter=use_adapter, adapter_ratio=adapter_ratio)
        elif backbone == 'imagenet_resnet50':
            self.encoder = models.resnet50(num_classes=num_cls)
        elif backbone == 'cnn':
            self.encoder = CNN(num_classes=num_cls, mlp=mlp)
        
    def forward(self, x):
        x = self.encoder(x)
        return x


class AdapterSimSiam2DLearner:
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
        torch.cuda.set_device(rank)
        
        # Model initialization
        net = SimSiamNet(self.cfg.backbone, self.cfg.out_dim, self.cfg.pred_dim, self.cfg.pretrain_mlp, self.cfg.use_adapter, self.cfg.adapter_ratio)
        net.cuda()
        if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
            cls_net = SimSiamClassifier(self.cfg.backbone, self.cfg.n_way, self.cfg.finetune_mlp, self.cfg.use_adapter, self.cfg.adapter_ratio)
            cls_net.cuda()
        
        # Setting Distributed Data Parallel configuration
        if world_size > 1:
            dist.init_process_group(backend='nccl', init_method=self.cfg.dist_url,
                                    world_size=world_size, rank=rank)
            
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = nn.parallel.DistributedDataParallel(net, device_ids=[rank],
                                                      find_unused_parameters=True)
            if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
                cls_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(cls_net)
                cls_net = nn.parallel.DistributedDataParallel(cls_net, device_ids=[rank],
                                                              find_unused_parameters=True)
            
            train_sampler = DistributedSampler(train_dataset)
            test_sampler = DistributedSampler(test_dataset)
            train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size//world_size,
                                      shuffle=False, sampler=train_sampler,
                                      num_workers=self.cfg.num_workers, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size//world_size,
                                     shuffle=False, sampler=test_sampler,
                                     num_workers=self.cfg.num_workers, drop_last=True)
            meta_train_dataset = Subset(train_dataset, list(train_sampler))
            
            if len(val_dataset) > 0:
                val_sampler = DistributedSampler(val_dataset)
                val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size//world_size,
                                        shuffle=False, sampler=val_sampler,
                                        num_workers=self.cfg.num_workers, drop_last=True)
            self.write_log(rank, logs, "DDP is used for training - training {} instances for each worker".format(len(list(train_sampler))))
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                                      num_workers=self.cfg.num_workers, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                                     num_workers=self.cfg.num_workers, drop_last=True)
            meta_train_dataset = train_dataset
            
            if len(val_dataset) > 0:
                val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                                        num_workers=self.cfg.num_workers, drop_last=True)
            self.write_log(rank, logs, "Single GPU is used for training - training {} instances for each worker".format(len(train_dataset)))
        
        # Define criterion
        simsiam_criterion = nn.CosineSimilarity(dim=1).cuda()
        if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
            cls_criterion = nn.CrossEntropyLoss().cuda()
        
        if self.cfg.mode == 'pretrain':
            if os.path.isfile(self.cfg.pretrained):
                loc = 'cuda:{}'.format(rank)
                state = torch.load(self.cfg.pretrained, map_location=loc)['state_dict']
                
                new_state = {}
                for k, v in state.items():
                    if k.startswith('module.'):
                        k = k.replace('module.', '')
                    new_state[k] = v
                    
                msg = net.load_state_dict(new_state, strict=False)
                self.write_log(rank, logs, "Loading pretrained model from checkpoint - {}".format(self.cfg.pretrained))
                self.write_log(rank, logs, f"missing keys: {msg.missing_keys}")
            else:
                self.write_log(rank, logs, "No checkpoint found at '{}'".format(self.cfg.pretrained))
        
        # For finetuning, load pretrained model
        if self.cfg.mode == 'finetune':
            if os.path.isfile(self.cfg.pretrained):
                loc = 'cuda:{}'.format(rank)
                state = torch.load(self.cfg.pretrained, map_location=loc)['state_dict']
                
                new_state = {}
                for k, v in state.items():
                    if k.startswith('module.'):
                        k = k.replace('module.', '')
                    new_state[k] = v
                    
                msg = net.load_state_dict(new_state, strict=False)
                self.write_log(rank, logs, "Loading pretrained model from checkpoint - {}".format(self.cfg.pretrained))
                self.write_log(rank, logs, f"missing keys: {msg.missing_keys}")
            else:
                self.write_log(rank, logs, "No checkpoint found at '{}'".format(self.cfg.pretrained))
                
                
            _IMAGENET_SIZE = (224, 224)
            _IMAGENET_MEAN = [0.485, 0.456, 0.406]
            _IMAGENET_STDDEV = [0.229, 0.224, 0.225]
            pre_transform = transforms.Compose([
                # transforms.Resize(_IMAGENET_SIZE)
                transforms.RandomResizedCrop(_IMAGENET_SIZE)
            ])
            post_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN,
                                    std=_IMAGENET_STDDEV)
            ])
                
            # def get_features(batch):
            #     x1 = []
            #     x2 = []
            #     for x in batch:
            #         x1_elem = post_transform(pre_transform(x[0][0]))
            #         x2_elem = post_transform(augs(pre_transform(x[0][0])))
            #         x1.append(x1_elem)
            #         x2.append(x2_elem)
            #     return torch.stack(x1).cuda(), torch.stack(x2).cuda(), y
            
            # support_loader = DataLoader(supports[idx], batch_size=len(supports[idx]), collate_fn=get_features)
            
            def get_features(batch):
                x1 = []
                x2 = []
                y = []
                for x in batch:
                    x1.append(x[0][1])
                    x2.append(x[0][2])
                    y.append(x[1])
                return torch.stack(x1).cuda(), torch.stack(x2).cuda(), y
            
            support_loader = DataLoader(meta_train_dataset, batch_size=len(meta_train_dataset), collate_fn=get_features)
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), collate_fn=get_features, shuffle=True)
            net.encoder.use_adapter = False
            self.write_log(rank, logs, "simsiam evaluation on val dataset")
            self.meta_eval(rank, net, val_loader, simsiam_criterion, logs)
            self.write_log(rank, logs, "simsiam evaluation on train dataset")
            self.meta_eval(rank, net, support_loader, simsiam_criterion, logs)
            
            net.encoder.use_adapter = self.cfg.use_adapter
            self.write_log(rank, logs, "simsiam evaluation on val dataset")
            self.meta_eval(rank, net, val_loader, simsiam_criterion, logs)
            self.write_log(rank, logs, "simsiam evaluation on train dataset")
            self.meta_eval(rank, net, support_loader, simsiam_criterion, logs)
            if self.cfg.use_adapter:
                self.write_log(rank, logs, "Performing domain adaptation")
                if world_size > 1:
                    train_sampler.set_epoch(0)
                # self.meta_train(rank, net, support_loader, simsiam_criterion, logs)
                # s_x1, s_x2, y = next(iter(support_loader))
                simsiam_criterion = nn.CosineSimilarity(dim=1).cuda()
                for name, p in net.named_parameters():
                    if 'vars' in name: p.requires_grad = True
                    else: p.requires_grad = False
                net.eval()
                # fast_weights = self.meta_train(rank, net, s_x1, s_x2, y, simsiam_criterion, self.cfg.log_steps, logs)
                for name, p in net.named_parameters():
                    if 'vars' in name:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                fast_weights = self.meta_train(rank, net, support_loader, simsiam_criterion, self.cfg.log_steps, logs)
                msg = net.load_state_dict({'encoder.vars.0': fast_weights[0], 'encoder.vars.1': fast_weights[1]}, strict=False)
            
            self.write_log(rank, logs, "simsiam evaluation on val dataset")
            self.meta_eval(rank, net, val_loader, simsiam_criterion, logs)
            self.write_log(rank, logs, "simsiam evaluation on train dataset")
            self.meta_eval(rank, net, support_loader, simsiam_criterion, logs)
            self.write_log(rank, logs, "Loading encoder parameters to classifier")
            enc_dict = {}
            for i, (k, v) in enumerate(cls_net.state_dict().items()):
                if not 'fc' in k:
                    enc_dict[k] = net.state_dict()[k]
                else:
                    enc_dict[k] = v
            msg = cls_net.load_state_dict(enc_dict, strict=True)
            self.write_log(rank, logs, f"missing keys: {msg.missing_keys}")
            net = cls_net
            
            # Freezing the encoder
            if self.cfg.freeze:
                self.write_log(rank, logs, "Freezing the encoder")
                for name, param in net.named_parameters():
                    if not 'fc' in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
        
        # parameters = [p for n, p in net.named_parameters() if p.requires_grad]
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
            self.write_log(rank, logs, "Loading checkpoint from '{}'".format(self.cfg.resume))
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
            self.validate_pretrain(rank, net, test_loader, simsiam_criterion, logs)
        elif self.cfg.mode == 'eval_finetune':
            self.validate_finetune(rank, net, test_loader, simsiam_criterion, logs)
        else:
            if self.cfg.mode == 'pretrain':
                self.write_log(rank, logs, "Getting indices by domain...")
                if world_size > 1:
                    meta_train_indices = meta_train_dataset.indices
                else:
                    meta_train_indices = list(range(len(meta_train_dataset)))
                meta_train_indices = sorted(meta_train_indices)
                indices_by_domain = train_dataset.get_indices_by_domain()
                sampled_indices_by_domain = defaultdict(list)
                for k, v in indices_by_domain.items():
                    v_min = v[0]
                    v_max = v[-1]
                    indices = []
                    for idx in meta_train_indices:
                        if idx < v_min or idx > v_max: break
                        indices.append(idx)
                    meta_train_indices = meta_train_indices[len(indices):]
                    sampled_indices_by_domain[k] = indices
                
            self.write_log(rank, logs, "Start training")
            
            if self.cfg.mode == 'pretrain':
                for name, p in net.named_parameters():
                    if 'vars' in name:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
                if self.cfg.optimizer == 'sgd':
                    optimizer = torch.optim.SGD(parameters, self.cfg.lr,
                                                momentum=self.cfg.momentum,
                                                weight_decay=self.cfg.wd)
                elif self.cfg.optimizer == 'adam':
                    optimizer = torch.optim.Adam(parameters, self.cfg.lr,
                                                weight_decay=self.cfg.wd)
            
            for epoch in tqdm(range(self.cfg.start_epoch, self.cfg.epochs)):
                if world_size > 1:
                    train_sampler.set_epoch(epoch)
                    if len(val_dataset) > 0:
                        val_sampler.set_epoch(epoch)
                    test_sampler.set_epoch(epoch)
                
                if self.cfg.mode == 'pretrain':
                    supports, queries = self.gen_random_tasks(meta_train_dataset, self.cfg.task_size, self.cfg.num_task)
                    self.pretrain(rank, net, supports, queries, simsiam_criterion, optimizer, epoch, self.cfg.epochs, logs)
                    
                elif self.cfg.mode == 'finetune':
                    self.finetune(rank, net, train_loader, cls_criterion, optimizer, epoch, self.cfg.epochs, logs)
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
                self.validate_finetune(rank, net, test_loader, cls_criterion, logs)
    
    def gen_per_domain_tasks(self, dataset, indices_by_domain, task_size, num_task):
        domains = list(indices_by_domain.keys())
        supports = []
        with torch.no_grad():
            for i in range(num_task):
                domain = random.choice(domains)
                sample_indices = indices_by_domain[domain]
                random.shuffle(sample_indices)
                support = Subset(dataset, sample_indices[:task_size])
                if len(support) < task_size:
                    support = support*(task_size//len(support))+support[:task_size%len(support)]
                supports.append(support)
        
        return supports
    
    def gen_random_tasks(self, dataset, task_size, num_task):
        supports = []
        queries = []
        with torch.no_grad():
            for _ in range(num_task):
                indices = list(range(len(dataset)))
                random.shuffle(indices)
                support = Subset(dataset, indices[:task_size])
                supports.append(support)
                query = Subset(dataset, indices[task_size:task_size*2])
                queries.append(query)
        
        return supports, queries
    
    def get_optimizer(self, net, state=None):
        for name, param in net.named_parameters():
            if 'adapter' in name:# or 'predictor' in name or 'fc' in name:
                param.requires_grad = True
                print(name)
            else:
                param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
        print(len(parameters))
        # print the parameter names to be trained
        optimizer = torch.optim.Adam(parameters, lr=self.cfg.meta_lr)
        # if self.cfg.optimizer == 'sgd':
        # optimizer = torch.optim.SGD(parameters, self.cfg.meta_lr,
        #                             momentum=self.cfg.momentum,
        #                             weight_decay=self.cfg.wd)
        if state is not None:
            optimizer.load_state_dict(state)
        return optimizer
    
    def pretrain(self, rank, net, supports, queries, criterion, optimizer, epoch, num_epochs, logs):
        net.eval()
        # net.train()
        net.zero_grad()
        
        _IMAGENET_SIZE = (224, 224)
        _IMAGENET_MEAN = [0.485, 0.456, 0.406]
        _IMAGENET_STDDEV = [0.229, 0.224, 0.225]
        pre_transform = transforms.Compose([
            # transforms.Resize(_IMAGENET_SIZE)
            transforms.RandomResizedCrop(_IMAGENET_SIZE)
        ])
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN,
                                 std=_IMAGENET_STDDEV)
        ])
        
        losses = []
        for idx in range(len(supports)):
            augs = self.get_random_augmentations()
            def get_features(batch):
                x1 = []
                x2 = []
                for x in batch:
                    x1_elem = post_transform(pre_transform(x[0]))
                    x2_elem = post_transform(augs(pre_transform(x[0])))
                    x1.append(x1_elem)
                    x2.append(x2_elem)
                return torch.stack(x1).cuda(), torch.stack(x2).cuda()
            query_loader = DataLoader(queries[idx], batch_size=len(queries[idx]), collate_fn=get_features)
            q_x1, q_x2 = next(iter(query_loader))
            # p1, p2, z1, z2 = net(q_x1, q_x2, vars=fast_weights)
            # loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            # losses.append(loss)
            logits, targets = net(q_x1, q_x2, None)
            criterion = nn.CrossEntropyLoss().cuda()
            loss = criterion(logits, targets)
            acc = (logits.argmax(dim=1) == targets).float().mean()
            
            if idx % self.cfg.log_freq == 0 and rank == 0:
                log = f'Epoch [{epoch+1}/{num_epochs}]-({idx}/{len(supports)}) '
                log += f'-------Query Loss: {loss.item():.4f}\tacc: {acc.item():.2f}'
                logs.append(log)
                print(log)
            support_loader = DataLoader(supports[idx], batch_size=len(supports[idx]), collate_fn=get_features)
            s_x1, s_x2 = next(iter(support_loader))
            fast_weights = self.meta_train(rank, net, s_x1, s_x2, None, criterion, self.cfg.log_steps, logs)
            
            query_loader = DataLoader(queries[idx], batch_size=len(queries[idx]), collate_fn=get_features)
            q_x1, q_x2 = next(iter(query_loader))
            # p1, p2, z1, z2 = net(q_x1, q_x2, vars=fast_weights)
            # loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            # losses.append(loss)
            logits, targets = net(q_x1, q_x2, None, vars=fast_weights)
            criterion = nn.CrossEntropyLoss().cuda()
            loss = criterion(logits, targets)
            acc = (logits.argmax(dim=1) == targets).float().mean()
            losses.append(loss)
            
            if idx % self.cfg.log_freq == 0 and rank == 0:
                log = f'Epoch [{epoch+1}/{num_epochs}]-({idx}/{len(supports)}) '
                log += f'-------Query Loss: {loss.item():.4f}\tacc: {acc.item():.2f}'
                logs.append(log)
                print(log)
        
        losses = torch.stack(losses, dim=0)
        loss = torch.sum(losses)/len(supports)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # def meta_train(self, rank, net, x1, x2, y, criterion, log_steps=False, logs=None):
    def meta_train(self, rank, net, loader, criterion, log_steps=False, logs=None):
        net.eval()
        fast_weights = list(net.encoder.vars)
        for step in range(self.cfg.task_steps):
            for x1, x2, y in loader:
                # p1, p2, z1, z2 = net(x1, x2, vars=fast_weights)
                # loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                logits, targets = net(x1, x2, y, vars=fast_weights)
                criterion = nn.CrossEntropyLoss().cuda()
                loss = criterion(logits, targets)
                acc = (logits.argmax(dim=1) == targets).float().mean()
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
                    # std = torch.std(torch.cat((F.normalize(z1, dim=1), F.normalize(z2, dim=1)), dim=0))
                    log = f'\tStep [{step}/{self.cfg.task_steps}] Loss: {loss.item():.4f}\tacc: {acc.item():.2f}'#\tstd: {std.item():.4f}'
                    logs.append(log)
                    print(log)
        
        return fast_weights
    
    def meta_eval(self, rank, net, val_loader, criterion, logs):
        net.eval()
        with torch.no_grad():
            val_z1s = []
            val_z2s = []
            val_ys = []
            losses = []
            
            if self.cfg.adapt_algorithm == 'simsiam':
                for inner_step, (val_x1, val_x2, val_y) in enumerate(val_loader):
                    # val_p1, val_p2, val_z1, val_z2 = net(val_x1, val_x2)
                    # val_z1s.append(val_z1)
                    # val_z2s.append(val_z2)
                    logits, targets = net(val_x1, val_x2, val_y)
                    criterion = nn.CrossEntropyLoss().cuda()
                    loss = criterion(logits, targets)
                    acc = (logits.argmax(dim=1) == targets).float().mean()
                    val_ys += val_y
                    # loss = -(criterion(val_p1, val_z2).mean() + criterion(val_p2, val_z1).mean()) * 0.5
                    losses.append(loss)
                
                # val_z1 = torch.cat(val_z1s, dim=0)
                # val_z2 = torch.cat(val_z2s, dim=0)
                loss = sum(losses)/len(losses)
                # val_len = len(val_z1)
                if self.cfg.log_steps:
                    # KNNCls = KNN(n_neighbors=5)
                    # KNNCls.fit(val_z1.detach().cpu().numpy()[:val_len//2], y=val_ys[:val_len//2])
                    # KNN_pred = KNNCls.predict(val_z1.detach().cpu().numpy()[val_len//2:])
                    # KNN_acc = np.mean(KNN_pred == val_ys[val_len//2:])*100
                    # std = torch.std(torch.cat((F.normalize(val_z1, dim=1), F.normalize(val_z2, dim=1)), dim=0))
                    self.write_log(rank, logs, f'\tStep [{inner_step}/{self.cfg.inner_steps}] Loss: {loss.item():.4f} Acc: {acc.item()*100:.2f}')#\tStd: {std.item():.4f}\tKNN Acc: {KNN_acc:.2f}%')
            elif self.cfg.adapt_algorithm == 'simclr':
                criterion = nn.CrossEntropyLoss().cuda()
                for inner_step, (val_x1, val_x2, val_y) in enumerate(val_loader):
                    logits, targets = net(val_x1, val_x2, val_y)
                    loss = criterion(logits, targets)
                    losses.append(loss)
                
                loss = sum(losses)/len(losses)
                if self.cfg.log_steps:
                    acc = (logits.argmax(dim=1) == targets).float().mean()
                    self.write_log(rank, logs, f'\tStep [{inner_step}/{self.cfg.inner_steps}] Loss: {loss.item():.4f} Acc: {acc.item()*100:.2f}')
            
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
    
    def get_random_augmentations(self):
        # List of available augmentations
        augmentations = [
            transforms.RandomResizedCrop,
            transforms.RandomHorizontalFlip,
            transforms.RandomVerticalFlip, 
            transforms.RandomRotation,
            
            transforms.ColorJitter,
            transforms.GaussianBlur,
            transforms.RandomInvert,
            transforms.RandomPerspective,
            transforms.RandomPosterize,
            transforms.RandomGrayscale,
            transforms.RandomAdjustSharpness,
            transforms.RandomSolarize,
            transforms.ElasticTransform,
        ]

        # Randomly select two augmentations
        selected_augmentations = random.sample(augmentations, 3)

        # Define random parameters for the selected augmentations
        augmentation_params = {}
        for augmentation in selected_augmentations:
            if augmentation == transforms.RandomResizedCrop:
                augmentation_params[augmentation] = {
                    "size": 224,
                    "scale": (random.uniform(0.5, 0.8), 0.8)
                }
            elif augmentation in [transforms.RandomHorizontalFlip,
                                  transforms.RandomVerticalFlip,
                                  transforms.RandomInvert,
                                  transforms.RandomGrayscale]:
                # Set the probability to 1.0 (always apply the flip)
                augmentation_params[augmentation] = {"p": 1.0}
            elif augmentation == transforms.RandomRotation:
                augmentation_params[augmentation] = {
                    "degrees": random.uniform(0, 180),  # Define random rotation degrees
                }
            elif augmentation == transforms.ColorJitter:
                augmentation_params[augmentation] = {
                    "brightness": random.uniform(0.1, 0.5),
                    "contrast": random.uniform(0.1, 0.5),
                    "saturation": random.uniform(0.1, 0.5),
                    "hue": random.uniform(0.1, 0.5),
                }
            elif augmentation == transforms.RandomPerspective:
                augmentation_params[augmentation] = {
                    "distortion_scale": random.uniform(0.1, 0.5),
                    "p": 1.0,
                }
            elif augmentation == transforms.RandomPosterize:
                augmentation_params[augmentation] = {
                    "bits": random.randint(4, 8),
                    "p": 1.0,
                }
            elif augmentation == transforms.RandomSolarize:
                augmentation_params[augmentation] = {
                    "threshold": random.uniform(0, 256),
                    "p": 1.0,
                }
            elif augmentation == transforms.RandomAdjustSharpness:
                augmentation_params[augmentation] = {
                    "sharpness_factor": random.uniform(0, 2),
                    "p": 1.0,
                }
            elif augmentation == transforms.ElasticTransform:
                augmentation_params[augmentation] = {
                    "alpha": random.uniform(1.0, 10.0),
                    "sigma": random.uniform(10.0, 50.0),
                }
            elif augmentation == transforms.GaussianBlur:
                augmentation_params[augmentation] = {
                    "kernel_size": random.choice([3, 5, 7, 9]),
                    "sigma": random.uniform(0.1, 2.0),
                }
            # Add more conditions for other augmentations as needed

        # Create a Compose transform with the selected augmentations and parameters
        composed_transform = transforms.Compose([
            # augmentation(**augmentation_params[augmentation]) for augmentation in selected_augmentations
            RandAugment()
        ])

        return composed_transform

    
    def finetune(self, rank, net, train_loader, criterion, optimizer, epoch, num_epochs, logs):
        # if self.cfg.freeze: net.eval()
        # else: net.train()
        # net.encoder.fc.train()
        net.eval()
        
        for batch_idx, data in enumerate(train_loader):
            features = data[0][0].cuda()
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
                f1, recall, precision = self.scores(total_logits, total_targets)
            total_loss /= len(val_loader)
            
            if rank == 0:
                log = f'[Finetune] Validation Loss: {total_loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
                log += f', F1: {f1.item():.2f}, Recall: {recall.item():.2f}, Precision: {precision.item():.2f}'
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


# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self):
        self.n = 5
        self.augment_list = augment_list()

    def __call__(self, img):
        self.ops = random.choices(self.augment_list, k=self.n)
        ops = self.ops
        self.m = random.uniform(0, 30)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img