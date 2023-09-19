import os
import random
import numpy as np

import clip
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

from torch.autograd import Variable
from torch.utils.data import DataLoader, DistributedSampler, Subset

from net.resnet import ResNet18, ModifiedResNet50, ResNet18_meta, ResNet50
from net.convnetDigit5 import CNN

from tqdm import tqdm

class Predictor(nn.Module):
    def __init__(self, dim, pred_dim=1):
        super(Predictor, self).__init__()
        self.layer = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                   nn.BatchNorm1d(pred_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(pred_dim, dim))
    
    def forward(self, x):
        x = self.layer(x)
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
            self.encoder.fc = nn.Sequential(self.encoder.fc, nn.BatchNorm1d(out_dim, affine=False))
        
        if backbone == 'imagenet_resnet50':
            self.predictor = nn.Sequential(nn.Linear(out_dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, out_dim)) # output layer
        else:
            self.predictor = Predictor(dim=out_dim, pred_dim=pred_dim)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        p1 = self.predictor(z1)
        z2 = self.encoder(x2)
        p2 = self.predictor(z2)
        
        return p1, p2, z1.detach(), z2.detach()


class SimSiamClassifier(nn.Module):
    def __init__(self, backbone, num_cls, mlp=True, adapter=False):
        super(SimSiamClassifier, self).__init__()
        self.encoder = None
        if backbone == 'resnet18':
            self.encoder = ResNet18(num_classes=num_cls, mlp=mlp)
        elif backbone == 'resnet50':
            self.encoder = ModifiedResNet50(num_classes=num_cls, mlp=mlp, adapter=adapter)
        elif backbone == 'imagenet_resnet50':
            self.encoder = models.resnet50(num_classes=num_cls)
        elif backbone == 'cnn':
            self.encoder = CNN(num_classes=num_cls, mlp=mlp)
        
    def forward(self, x):
        x = self.encoder(x)
        return x


class ReptileSimSiam2DLearner:
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
        net = SimSiamNet(self.cfg.backbone, self.cfg.out_dim, self.cfg.pred_dim, self.cfg.pretrain_mlp)
        net.cuda()
        if self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
            cls_net = SimSiamClassifier(self.cfg.backbone, self.cfg.n_way, self.cfg.finetune_mlp)
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
                
                new_state = {}
                for k, v in state.items():
                    if k.startswith('module.'):
                        k = k.replace('module.', '')
                    new_state[k] = v
                    
                msg = net.load_state_dict(new_state, strict=True)
                self.write_log(rank, logs, "Loading pretrained model from checkpoint - {}".format(self.cfg.pretrained))
                self.write_log(rank, logs, f"missing keys: {msg.missing_keys}")
            else:
                self.write_log(rank, logs, "No checkpoint found at '{}'".format(self.cfg.pretrained))
            
            def get_features(batch):
                x1 = []
                x2 = []
                y = []
                for x in batch:
                    x1.append(x[0][1])
                    x2.append(x[0][2])
                    y.append(x[1])
                return torch.stack(x1).cuda(), torch.stack(x2).cuda(), y
            
            # batch_indices = [torch.randint(len(meta_train_dataset), size=(self.cfg.inner_batch_size,)) for _ in range(self.cfg.inner_steps)]
            batch_indices = [list(range(len(meta_train_dataset))) for _ in range(self.cfg.inner_steps)]
            support_loader = DataLoader(meta_train_dataset, batch_sampler=batch_indices, collate_fn=get_features)
            
            if self.cfg.domain_adaptation:
                self.write_log(rank, logs, "Performing domain adaptation")
                if world_size > 1:
                    train_sampler.set_epoch(0)
                self.meta_train(rank, net, support_loader, simsiam_criterion, logs)
            
            # batch_indices = [torch.randint(len(val_dataset), size=(self.cfg.inner_batch_size,)) for _ in range(self.cfg.inner_steps)]
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), collate_fn=get_features, shuffle=True)
            self.meta_eval(rank, net, val_loader, simsiam_criterion, logs)
            self.write_log(rank, logs, "Loading encoder parameters to classifier")
            enc_dict = {}
            for i, (k, v) in enumerate(cls_net.state_dict().items()):
                if not 'fc' in k:
                    enc_dict[k] = net.state_dict()[k]
            msg = cls_net.load_state_dict(enc_dict, strict=False)
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
            for epoch in tqdm(range(self.cfg.start_epoch, self.cfg.epochs)):
                if world_size > 1:
                    train_sampler.set_epoch(epoch)
                    if len(val_dataset) > 0:
                        val_sampler.set_epoch(epoch)
                    test_sampler.set_epoch(epoch)
                
                if self.cfg.mode == 'pretrain':
                    supports = self.gen_per_domain_tasks(
                        train_dataset, sampled_indices_by_domain, self.cfg.task_size, self.cfg.num_task//world_size)
                    rand_supports = self.gen_random_tasks(
                        meta_train_dataset, self.cfg.task_size, self.cfg.multi_cond_num_task//world_size)
                    supports = supports + rand_supports
                    self.pretrain(rank, net, supports, simsiam_criterion, epoch, self.cfg.epochs, logs)
                    
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
        with torch.no_grad():
            for _ in range(num_task):
                indices = list(range(len(dataset)))
                random.shuffle(indices)
                support = Subset(dataset, indices[:task_size])
                supports.append(support)
        
        return supports
    
    def get_optimizer(self, net, state=None):
        optimizer = torch.optim.Adam(net.parameters(), lr=self.cfg.meta_lr)
        # if self.cfg.optimizer == 'sgd':
        #     optimizer = torch.optim.SGD(net.parameters(), self.cfg.lr,
        #                                 momentum=self.cfg.momentum,
        #                                 weight_decay=self.cfg.wd)
        if state is not None:
            optimizer.load_state_dict(state)
        return optimizer
    
    def pretrain(self, rank, net, supports, criterion, epoch, num_epochs, logs):
        num_tasks = len(supports)
        
        def get_features(batch):
            x1, x2, y = [], [], []
            for x in batch:
                x1.append(x[0][0])
                x2.append(x[0][1])
                y.append(x[1])
            return torch.stack(x1).cuda(), torch.stack(x2).cuda(), y
        
        old_weights = {}
        for name in net.state_dict():
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name: continue
            old_weights[name] = net.state_dict()[name]
        weight_updates = {name: 0 for name in old_weights}
        
        losses = []
        for task_idx in range(num_tasks):
            support = supports[task_idx]
            # batch_indices = [torch.randint(len(support), size=(self.cfg.inner_batch_size,)) for _ in range(self.cfg.inner_steps)]
            batch_indices = [list(range(len(support))) for _ in range(self.cfg.inner_steps)]
            support_loader = DataLoader(support, batch_sampler=batch_indices, collate_fn=get_features)
            
            loss = self.meta_train(rank, net, support_loader, criterion, logs)
            losses.append(loss)
            self.write_log(rank, logs, f'Task({task_idx}) Loss: {loss:.4f}')
             
            weight_updates = {name: weight_updates[name]+(net.state_dict()[name]-old_weights[name]) for name in old_weights}
            net.load_state_dict(old_weights, strict=False)
        
        frac_done = epoch/self.cfg.epochs
        epsilone = frac_done * self.cfg.epsilone_final + (1-frac_done) * self.cfg.epsilone_start
        net.load_state_dict({name :
            old_weights[name]+weight_updates[name]*epsilone/num_tasks
            for name in old_weights}, strict=False)

        if task_idx % self.cfg.log_freq == 0 and rank == 0:
            self.write_log(rank, logs, f'Epoch [{epoch+1}/{num_epochs}] Loss: {sum(losses)/len(losses):.4f}')
    
    def meta_train(self, rank, net, support_loader, criterion, logs):
        net.train()
        # net.eval()
        # net.zero_grad()
        optimizer = self.get_optimizer(net)
        for inner_step, (s_x1, s_x2, y) in enumerate(support_loader):
            p1, p2, z1, z2 = net(s_x1, s_x2)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if self.cfg.log_steps:
                KNNCls = KNN(n_neighbors=3)
                KNNCls.fit(z1.detach().cpu().numpy(), y=y)
                KNN_pred = KNNCls.predict(z2.detach().cpu().numpy())
                KNN_acc = np.mean(KNN_pred == y)*100
                std = torch.std(torch.cat((z1, z2), dim=0))
                self.write_log(rank, logs, f'\tStep [{inner_step}/{self.cfg.inner_steps}] Loss: {loss.item():.4f}\tStd: {std.item():.4f}\tKNN Acc: {KNN_acc:.2f}%')
        
        return loss.item()
    
    def meta_eval(self, rank, net, val_loader, criterion, logs):
        net.eval()
        with torch.no_grad():
            val_z1s = []
            val_z2s = []
            val_ys = []
            losses = []
            for inner_step, (val_x1, val_x2, val_y) in enumerate(val_loader):
                val_p1, val_p2, val_z1, val_z2 = net(val_x1, val_x2)
                val_z1s.append(val_z1)
                val_z2s.append(val_z2)
                val_ys += val_y
                loss = -(criterion(val_p1, val_z2).mean() + criterion(val_p2, val_z1).mean()) * 0.5
                losses.append(loss)
                
            val_z1 = torch.cat(val_z1s, dim=0)
            val_z2 = torch.cat(val_z2s, dim=0)
            loss = sum(losses)/len(losses)
            val_len = len(val_z1)
            if self.cfg.log_steps:
                KNNCls = KNN(n_neighbors=5)
                KNNCls.fit(val_z1.detach().cpu().numpy()[:val_len//2], y=val_ys[:val_len//2])
                KNN_pred = KNNCls.predict(val_z1.detach().cpu().numpy()[val_len//2:])
                KNN_acc = np.mean(KNN_pred == val_ys[val_len//2:])*100
                std = torch.std(torch.cat((val_z1, val_z2), dim=0))
                self.write_log(rank, logs, f'Pre-trained task loss: {loss.item():.4f}\tStd: {std.item():.4f}\tKNN Acc: {KNN_acc:.2f}%')
        # assert(0)
            
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
