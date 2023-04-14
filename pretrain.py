#!/usr/bin/env python
# Code based on Facebook moco/simsiam implementation

# ignoring warnings
import warnings
warnings.filterwarnings("ignore", message="torch.distributed._all_gather_base is a private function and will be deprecated")

import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import core.loader
import core.builder
import core.transforms
import core.relnet
import core.resnet
import core.net.CPC

from data_loader.HHARDataset import HHARDataset

def run(config):
    cudnn.benchmark = True
    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if config.multiprocessing.dist_url == "env://" and config.multiprocessing.world_size == -1:
        config.multiprocessing.world_size = int(os.environ["WORLD_SIZE"])

    config.multiprocessing.distributed = config.multiprocessing.world_size > 1 \
        or config.multiprocessing.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print(f'cuda is_available: {torch.cuda.is_available()}')
    print(f'ngpus_per_node: {ngpus_per_node}')
    if config.multiprocessing.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.multiprocessing.world_size = ngpus_per_node * config.multiprocessing.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config)

def main_worker(gpu, ngpus_per_node, config):
    config.gpu = gpu
    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))
    
    # suppress printing if not master
    if config.multiprocessing.multiprocessing_distributed \
        and config.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if config.multiprocessing.distributed:
        if config.multiprocessing.dist_url == "env://" \
            and config.multiprocessing.rank == -1:
            config.multiprocessing.rank = int(os.environ["RANK"])
        if config.multiprocessing.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.multiprocessing.rank = config.multiprocessing.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.multiprocessing.dist_backend,
                                init_method=config.multiprocessing.dist_url,
                                world_size=config.multiprocessing.world_size,
                                rank=config.multiprocessing.rank)
        torch.distributed.barrier()
    
    # create model
    print("=> creating model '{}'".format(config.model.type))
    
    # TODO: Modulize model creation
    model = None
    if config.model.type == 'augcon':
        if config.model.d_encoder == None or config.model.d_encoder == 'identity':
            d_encoder = core.net.IdentityNet()
        elif config.model.d_encoder == 'convnet':
            d_encoder = core.net.ConvNet()
        else:
            assert 0, f"Network not supported"
            
        if config.model.r_encoder == 'convnet':
            r_encoder = core.net.ConvNet()
        elif config.model.r_encoder == 'resnet':
            r_encoder = core.net.ResNet()
        elif config.model.r_encoder == 'relnet':
            r_encoder = core.net.RelNet()
        else:
            assert 0, f"Network not supported"
            
        model = core.builder.AugCon(d_encoder, r_encoder, config.model.temp)
    
    elif config.model.type == 'cpc':
        encoder = core.net.CPC.Encoder(config.model.input_channels,
                                      config.model.z_dim)
        encoder = encoder.cuda(config.gpu)
        aggregator = core.net.CPC.Aggregator(config.model.num_blocks,
                                            config.model.num_filters)
        aggregator = aggregator.cuda(config.gpu)
        model = core.net.CPC.FuturePredictor(encoder, aggregator,
                                                config.model.z_dim,
                                                config.model.pred_steps,
                                                config.model.n_negatives,
                                                config.model.offset)
        
    elif config.model.type == 'moco':
        if config.model.d_encoder == 'convnet':
            d_encoder = core.net.ConvNet()
        elif config.model.d_encoder == 'resnet':
            d_encoder = core.net.ResNet()
        else:
            assert 0, f"Network not supported"
    
    elif config.model.type == 'simclr':
        if config.model.d_encoder == 'convnet':
            d_encoder = core.net.ConvNet()
        elif config.model.d_encoder == 'resnet':
            d_encoder = core.net.ResNet()
        else:
            assert 0, f"Network not supported"

    if config.multiprocessing.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.train.batch_size = int(config.train.batch_size / ngpus_per_node)
            config.multiprocessing.workers = int((config.multiprocessing.workers \
                + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[config.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    # TODO: Define new criterion for AugCon
    if config.train.criterion == 'cosinesimilarity':
        criterion = nn.CosineSimilarity(dim=1).cuda(config.gpu)
    elif config.train.criterion == 'crossentropy':
        criterion = nn.CrossEntropyLoss().cuda(config.gpu)
    else:
        assert 0, f"criterion not supported"

    if config.train.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    # TODO: Define new option for configurable optimizer
    if config.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(optim_params, config.train.lr,
                                    momentum=config.train.momentum,
                                    weight_decay=config.train.weight_decay)
    elif config.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(optim_params, config.train.lr,
                                     weight_decay=config.train.weight_decay)
    else:
        assert 0, f"optimizer not supported"

    # optionally resume from a checkpoint
    if config.train.resume:
        if os.path.isfile(config.train.resume):
            print("=> loading checkpoint '{}'".format(config.train.resume))
            if config.gpu is None:
                checkpoint = torch.load(config.train.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(config.gpu)
                checkpoint = torch.load(config.train.resume, map_location=loc)
            config.train.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(config.train.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config.train.resume))

    if config.data.name == 'hhar':
        dataset = HHARDataset(
            file=config.data.path,
            class_type=config.data.class_type,
            domain_type=config.data.domain_type,
            load_cache=config.data.load_cache,
            save_cache=config.data.save_cache,
            cache_path=config.data.cache_path,
            split_ratio=config.data.split_ratio,
            save_opposite=config.data.save_opposite,
            user=config.data.user,
            model=config.data.model,
            complementary=True
        )
    else:
        dataset = HHARDataset(
            file=config.data.path,
            class_type=config.data.class_type,
            domain_type=config.data.domain_type,
            load_cache=config.data.load_cache,
            save_cache=config.data.save_cache,
            cache_path=config.data.cache_path,
            split_ratio=config.data.split_ratio,
            save_opposite=config.data.save_opposite,
            user=config.data.user,
            model=config.data.model,
            complementary=True
        )
        
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset)-val_size-train_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    if config.multiprocessing.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
        
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=config.train.batch_size, shuffle=(train_sampler is None),
    #     num_workers=config.multiprocessing.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train.batch_size, shuffle=(train_sampler is None),
        num_workers=config.multiprocessing.workers, sampler=train_sampler, drop_last=True)

    print("start training...")
    for epoch in range(config.train.start_epoch, config.train.epochs):
        if config.multiprocessing.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, config)

        if not config.multiprocessing.multiprocessing_distributed \
                or (config.multiprocessing.multiprocessing_distributed
                    and config.multiprocessing.rank % ngpus_per_node == 0):
            save_checkpoint(config.train.save_dir, {
                'epoch': epoch + 1,
                'type': config.model.type,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, criterion, optimizer, epoch, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, x in enumerate(train_loader):
        feature = x[0].cuda(config.gpu)
        class_label = x[1].cuda(config.gpu)
        domain_label = x[2].cuda(config.gpu)
        
        logits, targets = model(feature)
        loss = criterion(logits, targets)

        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        losses.update(loss.item(), logits.size(0))
        top1.update(acc1[0], logits.size(0))
        top5.update(acc5[0], logits.size(0))
        losses.update(loss.item(), logits.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.train.print_freq == 0:
            progress.display(i)


def save_checkpoint(save_dir, state, is_best, filename='checkpoint.pth.tar'):
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate based on schedule"""
    lr = config.train.lr
    if config.train.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / config.train.epochs))
    else:  # stepwise lr schedule
        for milestone in config.train.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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