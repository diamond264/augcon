import os
import random
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

        labels = torch.arange(batch_size).cuda()
        masks = F.one_hot(torch.arange(batch_size), batch_size).cuda()
        # mask = ~torch.eye(batch_size, dtype=bool).cuda()

        logits_aa = torch.matmul(z, z.t())
        # logits_aa = logits_aa[mask].reshape(batch_size, batch_size-1)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(aug_z, aug_z.t())
        # logits_bb = logits_bb[mask].reshape(batch_size, batch_size-1)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(z, aug_z.t())

        logits = torch.cat([logits_ab, logits_aa, logits_bb], dim=1)
        # logits = logits_ab
        logits /= self.T
        # logits = F.pad(logits, (0, full_batch_size-logits.shape[1]), "constant", -LARGE_NUM)

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
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
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
        net = SimCLRNet(self.cfg.input_channels, self.cfg.z_dim, self.cfg.out_dim, self.cfg.T, True)
        if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
            cls_net = SimCLRClassifier(self.cfg.input_channels, self.cfg.z_dim, self.cfg.num_cls, self.cfg.mlp)
        
        # DDP setting
        if world_size > 1:
            dist.init_process_group(backend='nccl',
                                    init_method=self.cfg.dist_url,
                                    world_size=world_size,
                                    rank=rank)
            
            torch.cuda.set_device(rank)
            net.cuda()
            net = nn.parallel.DistributedDataParallel(net, device_ids=[rank], find_unused_parameters=True)
            
            train_sampler = DistributedSampler(train_dataset)
            meta_train_dataset = torch.utils.data.Subset(train_dataset, list(train_sampler))
            
            if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
                cls_net.cuda()
                cls_net = nn.parallel.DistributedDataParallel(cls_net, device_ids=[rank], find_unused_parameters=True)
                
                if len(val_dataset) > 0:
                    val_sampler = DistributedSampler(val_dataset)
                test_sampler = DistributedSampler(test_dataset)
                
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
            
            if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
                cls_net.cuda()
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
        
        indices_per_domain = self.split_per_domain(meta_train_dataset)
        # indices_per_domain = self.split_per_domain(train_dataset)
        
        # Define criterion
        if self.cfg.criterion == 'crossentropy':
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
                
                # for k, v in list(state.items()):
                #     if k.startswith('encoder.'):
                #         k = k[len('encoder.'):]
                #     if world_size > 1:
                #         k = 'module.' + k
                #     if k in net.state_dict().keys():
                #         state[k] = v
                new_state = {}
                if self.cfg.no_vars:
                    for i, (k, v) in enumerate(state.items()):
                        new_k = list(net.state_dict().keys())[i]
                        new_state[new_k] = v
                else:
                    new_state = state
                
                # print(state.keys())
                # print(net.state_dict().keys())
                # assert(0)
                
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
            
            # Meta-train the pretrained model for domain adaptation
            if self.cfg.domain_adaptation:
                if rank == 0:
                    log = "Perform domain adaptation step"
                    logs.append(log)
                    print(log)
                if world_size > 1:
                    train_sampler.set_epoch(0)
                net.train()
                net.zero_grad()

                shuffled_idx = torch.randperm(len(meta_train_dataset))
                meta_train_dataset = torch.utils.data.Subset(meta_train_dataset, shuffled_idx)

                if self.cfg.out_cls_neg_sampling:
                    enc_parameters = self.adapt(rank, net, meta_train_dataset, criterion, log_steps=True, logs=logs)
                    self.meta_eval(rank, net, test_dataset, criterion, enc_parameters, logs)
                else:
                    support = [e[1] for e in meta_train_dataset]
                    pos_support = [e[2] for e in meta_train_dataset]
                    support = torch.stack(support, dim=0).cuda()
                    pos_support = torch.stack(pos_support, dim=0).cuda()
                    enc_parameters = self.meta_train(rank, net, support, pos_support, criterion, log_steps=True, logs=logs)
                    # self.meta_eval(rank, net, test_dataset, criterion, enc_parameters, logs)
            else:
                enc_parameters = list(net.parameters())
            
            if rank == 0:
                log = "Loading encoder parameters to the classifier"
                logs.append(log)
                print(log)
            
            enc_dict = {}
            for idx, k in enumerate(list(net.state_dict().keys())):
                if not 'classifier' in k:
                    k_ = k.replace('encoder.', 'base_model.')
                    enc_dict[k_] = enc_parameters[idx]
            
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
                    print(name)
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
            # loss_best = 0
            for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
                if world_size > 1:
                    train_sampler.set_epoch(epoch)
                    if len(val_dataset) > 0:
                        val_sampler.set_epoch(epoch)
                    test_sampler.set_epoch(epoch)
                
                if self.cfg.mode == 'pretrain':
                    supports = []
                    queries = []
                    pos_supports = []
                    pos_queries = []
                    if self.cfg.task_per_domain:
                        supports, pos_supports, queries, pos_queries = self.gen_per_domain_tasks(
                            meta_train_dataset,
                            indices_per_domain,
                            self.cfg.task_size,
                            self.cfg.num_task
                        )
                    if self.cfg.multi_cond_num_task > 0:
                        multi_cond_supports, multi_cond_pos_supports, multi_cond_queries, multi_cond_pos_queries = self.gen_random_tasks(
                            meta_train_dataset,
                            self.cfg.task_size,
                            self.cfg.multi_cond_num_task
                        )
                        supports = supports+multi_cond_supports
                        queries = queries+multi_cond_queries
                        pos_supports = pos_supports+multi_cond_pos_supports
                        pos_queries = pos_queries+multi_cond_pos_queries

                    # print(f"Num task : {len(supports)}")
                    self.pretrain(rank, net, supports, pos_supports, queries, pos_queries, criterion, optimizer, epoch, self.cfg.epochs, logs)
                    # if len(val_dataset) > 0:
                    #     loss_ep = self.validate_pretrain(rank, net, val_loader, criterion, logs)
                    
                    # # Storing best epoch checkpoint and early stopping
                    # if rank == 0:
                    #     if loss_best == 0 or loss_ep < loss_best:
                    #         loss_best = loss_ep
                    #         esnum = 0
                    #         # Save best checkpoint
                    #         ckpt_dir = self.cfg.ckpt_dir
                    #         ckpt_filename = 'checkpoint_best.pth.tar'
                    #         ckpt_filename = os.path.join(ckpt_dir, ckpt_filename)
                    #         state_dict = net.state_dict()
                    #         if world_size > 1:
                    #             for k, v in list(state_dict.items()):
                    #                 if 'module.' in k:
                    #                     state_dict[k.replace('module.', '')] = v
                    #                     del state_dict[k]
                    #         self.save_checkpoint(ckpt_filename, epoch, state_dict, optimizer)
                    #     else:
                    #         esnum = esnum+1
                    #         # Early stop
                    #         if self.cfg.earlystop > 0 and esnum == self.cfg.earlystop:
                    #             log = "Early Stopped at best epoch {}".format(epoch-5)
                    #             logs.append(log)
                    #             print(log)
                    #             break
                    
                elif self.cfg.mode == 'finetune':
                    self.finetune(rank, net, train_loader, criterion, optimizer, epoch, self.cfg.epochs, logs)
                    if len(val_dataset) > 0:
                        self.validate_finetune(rank, net, val_loader, criterion, logs)
                
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
    
    def split_per_domain(self, dataset):
        indices_per_domain = defaultdict(list)
        for i, d in enumerate(dataset):
            indices_per_domain[d[4].item()].append(i)
        return indices_per_domain
    
    def split_per_class(self, dataset):
        indices_per_class = defaultdict(list)
        opp_indices_per_class = defaultdict(list)
        for i, d in enumerate(dataset):
            indices_per_class[d[3].item()].append(i)
        for cls, indices in indices_per_class.items():
            for i in range(len(dataset)):
                if i not in indices:
                    opp_indices_per_class[cls].append(i)
        return indices_per_class, opp_indices_per_class
    
    def gen_per_domain_tasks(self, dataset, indices_per_domain, task_size, num_task=None):
        supports = []
        queries = []
        pos_supports = []
        pos_queries = []
        
        with torch.no_grad():
            if num_task is None:
                for _, indices in indices_per_domain.items():
                    random.shuffle(indices)
                    support_ = torch.utils.data.Subset(dataset, indices[:task_size])
                    support = []
                    pos_support = []
                    for e in support_:
                        support.append(e[1])
                        pos_support.append(e[2])
                    support = torch.stack(support, dim=0)
                    pos_support = torch.stack(pos_support, dim=0)
                    
                    query_ = torch.utils.data.Subset(dataset, indices[task_size:2*task_size])
                    query = []
                    pos_query = []
                    for e in query_:
                        query.append(e[1])
                        pos_query.append(e[2])
                    query = torch.stack(query, dim=0)
                    pos_query = torch.stack(pos_query, dim=0)
                    
                    supports.append(support)
                    queries.append(query)
                    pos_supports.append(pos_support)
                    pos_queries.append(pos_query)
            else:
                for _ in range(num_task):
                    dom = random.choice(list(indices_per_domain.keys()))
                    indices = indices_per_domain[dom]
                    random.shuffle(indices)
                    support_ = torch.utils.data.Subset(dataset, indices[:task_size])
                    # support_ = torch.utils.data.Subset(dataset, indices[:len(indices)//2])
                    support = []
                    pos_support = []
                    for e in support_:
                        support.append(e[1])
                        pos_support.append(e[2])
                    if len(support) >= task_size:
                        support = support[:task_size]
                        pos_support = pos_support[:task_size]
                    else:
                        support = support * (task_size // len(support)) + support[:task_size % len(support)]
                        pos_support = pos_support * (task_size // len(pos_support)) + pos_support[:task_size % len(pos_support)]
                    support = torch.stack(support, dim=0)
                    pos_support = torch.stack(pos_support, dim=0)
                    
                    query_ = torch.utils.data.Subset(dataset, indices[task_size:2*task_size])
                    # query_ = torch.utils.data.Subset(dataset, indices[len(indices)//2:])
                    query = []
                    pos_query = []
                    for e in query_:
                        query.append(e[1])
                        pos_query.append(e[2])
                    if len(query) >= task_size:
                        query = query[:task_size]
                        pos_query = pos_query[:task_size]
                    else:
                        query = query * (task_size // len(query)) + query[:task_size % len(query)]
                        pos_query = pos_query * (task_size // len(pos_query)) + pos_query[:task_size % len(pos_query)]
                    query = torch.stack(query, dim=0)
                    pos_query = torch.stack(pos_query, dim=0)
                    
                    supports.append(support)
                    queries.append(query)
                    pos_supports.append(pos_support)
                    pos_queries.append(pos_query)
        
        return supports, pos_supports, queries, pos_queries

    def gen_random_tasks(self, dataset, task_size, num_task):
        supports = []
        queries = []
        pos_supports = []
        pos_queries = []
        with torch.no_grad():
            for _ in range(num_task):
                indices = list(range(len(dataset)))
                random.shuffle(indices)
                
                st = 0
                ed = task_size
                
                support_ = torch.utils.data.Subset(dataset, indices[st:ed])
                support = []
                pos_support = []
                for e in support_:
                    support.append(e[1])
                    pos_support.append(e[2])
                support = torch.stack(support, dim=0)
                pos_support = torch.stack(pos_support, dim=0)
                st += task_size
                ed += task_size
                
                query_ = torch.utils.data.Subset(dataset, indices[st:ed])
                query = []
                pos_query = []
                for e in query_:
                    query.append(e[1])
                    pos_query.append(e[2])
                query = torch.stack(query, dim=0)
                pos_query = torch.stack(pos_query, dim=0)
                st += task_size
                ed += task_size
                supports.append(support)
                queries.append(query)
                pos_supports.append(pos_support)
                pos_queries.append(pos_query)
            
        return supports, pos_supports, queries, pos_queries
    
    def pretrain(self, rank, net, supports, pos_supports, queries, pos_queries, criterion, optimizer, epoch, num_epochs, logs):
        net.train()
        net.zero_grad()
        
        q_losses = []
        for task_idx in range(len(supports)):
            support = supports[task_idx].cuda()
            pos_support = pos_supports[task_idx].cuda()
            query = queries[task_idx].cuda()
            pos_query = pos_queries[task_idx].cuda()
            
            fast_weights = self.meta_train(rank, net, support, pos_support, criterion, log_steps=self.cfg.log_meta_train, logs=logs)
            
            q_logits, q_targets = net(query, pos_query, fast_weights)
            q_loss = criterion(q_logits, q_targets)
            q_losses.append(q_loss)
            
            if task_idx % self.cfg.log_freq == 0:
                acc1, acc5 = self.accuracy(q_logits, q_targets, topk=(1, 5))
                log = f'Epoch [{epoch+1}/{num_epochs}]  \t({task_idx}/{len(supports)}) '
                log += f'Loss: {q_loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
                if rank == 0:
                    logs.append(log)
                    print(log)
        
        q_losses = torch.stack(q_losses, dim=0)
        loss = torch.sum(q_losses)
        # reg_term = torch.sum((q_losses - torch.mean(q_losses))**2)
        # loss += reg_term * self.cfg.reg_lambda
        loss = loss/len(supports)
        # log = f'Epoch [{epoch+1}/{num_epochs}] {loss.item():.4f}'
        # if rank == 0:
        #     logs.append(log)
        #     print(log)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def meta_train(self, rank, net, support, pos_support, criterion, log_steps=False, logs=None):
        fast_weights = list(net.parameters())
        for i in range(self.cfg.task_steps):
            s_logits, s_targets = net(support, pos_support, fast_weights)
            s_loss = criterion(s_logits, s_targets)
            grad = torch.autograd.grad(s_loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.cfg.task_lr * p[0], zip(grad, fast_weights)))
            
            if log_steps and rank == 0:
                acc1, acc5 = self.accuracy(s_logits, s_targets, topk=(1, 5))
                log = f'\tmeta-train [{i}/{self.cfg.task_steps}] Loss: {s_loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
                logs.append(log)
                print(log)
        
        return fast_weights
    
    def meta_eval(self, rank, net, test_dataset, criterion, fast_weights, logs):
        indices = list(range(len(test_dataset)))
        random.shuffle(indices)
        dataset = torch.utils.data.Subset(test_dataset, indices[:90])
        # dataset = test_dataset
        support = [e[0] for e in dataset]
        pos_support = [e[2] for e in dataset]
        support = torch.stack(support, dim=0).cuda()
        pos_support = torch.stack(pos_support, dim=0).cuda()
        
        logits, targets = net(support, pos_support, fast_weights)
        loss = criterion(logits, targets)
        
        
        train_dataset = test_dataset
        indices_per_class, opp_indices_per_class = self.split_per_class(train_dataset)
        full_batch_size = len(train_dataset)
        
        total_logits = []
        total_targets = []
        
        for cls, indices in indices_per_class.items():
            opp_indices = opp_indices_per_class[cls]
            
            support_ = torch.utils.data.Subset(train_dataset, indices)
            support = []
            pos_support = []
            for e in support_:
                support.append(e[0])
                pos_support.append(e[2])
            support = torch.stack(support, dim=0).cuda()
            pos_support = torch.stack(pos_support, dim=0).cuda()
            
            neg_support_ = torch.utils.data.Subset(train_dataset, opp_indices)
            neg_support = []
            neg_aug_support = []
            for e in neg_support_:
                neg_support.append(e[0])
                neg_aug_support.append(e[2])
            neg_support = torch.stack(neg_support, dim=0).cuda()
            neg_aug_support = torch.stack(neg_aug_support, dim=0).cuda()
            
            logits, targets = net.adapt(support, pos_support, neg_support, neg_aug_support, full_batch_size, fast_weights)
            total_logits.append(logits)
            total_targets.append(targets)
        
        logits = torch.cat(total_logits, dim=0)
        targets = torch.cat(total_targets, dim=0)
        loss = criterion(logits, targets)
        
        
        
        if rank == 0:
            acc1, acc5 = self.accuracy(logits, targets, topk=(1, 5))
            log = f'\tmeta-eval Loss: {loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
            logs.append(log)
            print(log)
    
    def adapt(self, rank, net, train_dataset, criterion, log_steps=False, logs=None):
        fast_weights = list(net.parameters())
        indices_per_class, opp_indices_per_class = self.split_per_class(train_dataset)
        full_batch_size = len(train_dataset)
        
        for i in range(self.cfg.task_steps):
            total_logits = []
            total_targets = []
            
            for cls, indices in indices_per_class.items():
                opp_indices = opp_indices_per_class[cls]
                
                support_ = torch.utils.data.Subset(train_dataset, indices)
                support = []
                pos_support = []
                for e in support_:
                    support.append(e[1])
                    pos_support.append(e[2])
                support = torch.stack(support, dim=0).cuda()
                pos_support = torch.stack(pos_support, dim=0).cuda()
                
                neg_support_ = torch.utils.data.Subset(train_dataset, opp_indices)
                neg_support = []
                neg_aug_support = []
                for e in neg_support_:
                    neg_support.append(e[1])
                    neg_aug_support.append(e[2])
                neg_support = torch.stack(neg_support, dim=0).cuda()
                neg_aug_support = torch.stack(neg_aug_support, dim=0).cuda()
                
                logits, targets = net.adapt(support, pos_support, neg_support, neg_aug_support, full_batch_size, fast_weights)
                total_logits.append(logits)
                total_targets.append(targets)
            
            total_logits = torch.cat(total_logits, dim=0)
            total_targets = torch.cat(total_targets, dim=0)
            loss = criterion(total_logits, total_targets)
            grad = torch.autograd.grad(loss, fast_weights)
            for g, p in zip(grad, fast_weights):
                g += self.cfg.task_wd * p
            fast_weights = list(map(lambda p: p[1] - self.cfg.task_lr * p[0], zip(grad, fast_weights)))
            
            if log_steps and rank == 0:
                acc1, acc5 = self.accuracy(total_logits, total_targets, topk=(1, 5))
                log = f'\tmeta-train [{i}/{self.cfg.task_steps}] Loss: {loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
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
                domains = data[3].cuda()
                
                if self.cfg.neg_per_domain:
                    all_logits = []
                    all_targets = []
                    for dom in self.all_domains:
                        dom_idx = torch.nonzero(domains == dom).squeeze()
                        if dom_idx.dim() == 0: dom_idx = dom_idx.unsqueeze(0)
                        if torch.numel(dom_idx):
                            dom_features = features[dom_idx]
                            dom_pos_features = pos_features[dom_idx]
                            logits, targets = net(dom_features, dom_pos_features, features.shape[0])
                            all_logits.append(logits)
                            all_targets.append(targets)
                    logits = torch.cat(all_logits, dim=0)
                    targets = torch.cat(all_targets, dim=0)
                else:
                    logits, targets = net(features, pos_features, features.shape[0])
                loss = criterion(logits, targets)
                total_loss += loss
            
            # if len(total_targets) > 0:
            #     total_targets = torch.cat(total_targets, dim=0)
            #     total_logits = torch.cat(total_logits, dim=0)
            #     acc1, acc5 = self.accuracy(total_logits, total_targets, topk=(1, 5))
            #     # f1, recall, precision = self.scores(total_logits, total_targets)
            total_loss /= len(val_loader)
            
            if rank == 0:
                log = f'[Pretrain] Validation Loss: {total_loss.item():.4f}'#, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
                logs.append(log)
                print(log)
            
            return total_loss.item()
    
    def finetune(self, rank, net, train_loader, criterion, optimizer, epoch, num_epochs, logs):
        net.eval()

        for batch_idx, data in enumerate(train_loader):
            features = data[0].cuda()
            targets = data[3].cuda()
            
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
                targets = data[3].cuda()
                
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
