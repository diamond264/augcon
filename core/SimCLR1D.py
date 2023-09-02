import os
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

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=24)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)

        self.conv3 = nn.Conv1d(64, z_dim, kernel_size=8)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)

        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.global_max_pooling(x)
        x = x.squeeze(-1)
        return x


class Classifier(nn.Module):
    def __init__(self, base_model, in_dim=96,
                 hidden_1=256, hidden_2=128, out_dim=50):
        super(Classifier, self).__init__()
        self.base_model = base_model
        self.fc1 = nn.Linear(in_dim, hidden_1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_2, out_dim)

    def forward(self, x):
        base_model_output = self.base_model(x)
        x = self.fc1(base_model_output)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


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

        if mlp:  # hack: brute-force replacement
            self.encoder = Classifier(self.encoder, in_dim=z_dim, out_dim=out_dim)

    def forward(self, feature, aug_feature, full_batch_size):
        # compute query features
        z = self.encoder(feature)  # queries: NxC
        z = nn.functional.normalize(z, dim=1)
            
        aug_z = self.encoder(aug_feature)
        aug_z = nn.functional.normalize(aug_z, dim=1)
        
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
        logits = F.pad(logits, (0, full_batch_size*3-logits.shape[1]), "constant", -LARGE_NUM)

        return logits, labels


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


class SimCLR1DLearner:
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
        if self.cfg.mode == 'pretrain' or self.cfg.mode == 'eval_pretrain':
            net = SimCLRNet(self.cfg.input_channels, self.cfg.z_dim, self.cfg.out_dim, self.cfg.T, self.cfg.mlp)
        elif self.cfg.mode == 'finetune' or self.cfg.mode == 'eval_finetune':
            net = SimCLRClassifier(self.cfg.input_channels, self.cfg.z_dim, self.cfg.num_cls, self.cfg.mlp)
        
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
        
        self.all_domains = self.split_per_domain(train_dataset)
        
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
                
                for k, v in list(state.items()):
                    if k.startswith('encoder.'):
                        k = k[len('encoder.'):]
                    if world_size > 1:
                        k = 'module.' + k
                    if k in net.state_dict().keys():
                        state[k] = v
                
                msg = net.load_state_dict(state, strict=False)
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
                    self.pretrain(rank, net, train_loader, criterion, optimizer, epoch, self.cfg.epochs, logs)
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
        domains = []
        for d in dataset:
            if d[3] not in domains:
                domains.append(d[3])
        domains = sorted(domains)
        return domains
    
    def pretrain(self, rank, net, train_loader, criterion, optimizer, epoch, num_epochs, logs):
        net.train()
        
        for batch_idx, data in enumerate(train_loader):
            features = data[0].cuda()
            pos_features = data[1].cuda()
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

            if batch_idx % self.cfg.log_freq == 0 and rank == 0:
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
        
        total_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                features = data[1].cuda()
                pos_features = data[2].cuda()
                domains = data[4].cuda()
                
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
        if self.cfg.freeze: net.eval()
        else: net.train()
        
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
                targets = data[2].cuda()
                
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
