#!/usr/bin/env python
# Code based on Facebook moco/simsiam implementation

'''
TODO:
- implement functionality to log configurations
- attach tensorboard
- add function to automatically parse augmentations
- replace BatchNorm to LayerNorm
'''

# ignoring warnings
import warnings
warnings.filterwarnings("ignore", message="torch.distributed._all_gather_base is a private function and will be deprecated")

import argparse
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

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def run(config):
    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if config.multiprocessing.gpu is not None:
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
        config.multiprocessing.world_size = ngpus_per_node * aconfig.multiprocessing.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, config=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config.multiprocessing.gpu, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    
    # create model
    # TODO: Need to implement AugCon class
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'resnet18':
        encoder = core.resnet.Encoder_res18()
        discriminator = core.resnet.Discriminator_res() 
        model = core.builder.AugCon(
            encoder, discriminator, args.temp)

    elif args.arch == 'relnet':
        encoder = core.relnet.CNNEncoder()
        discriminator = core.relnet.RelationNetwork()
        model = core.builder.AugCon(encoder, discriminator, args.temp)

    else:
        encoder = None
        discriminator = None
        model = None

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    # TODO: Define new criterion for AugCon
    # criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    # TODO: Define new option for configurable optimizer
    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    pre_process = [
        transforms.Resize((32, 32))
    ]

    post_process = [
        transforms.ToTensor(),
        normalize
    ]

    # EXAMPLE TRANSFORMATIONS
    base_transforms = [
        (core.transforms.ShearX, [-0.5, 0.5], 0.8),
        (core.transforms.ShearY, [-0.5, 0.5], 0.8),
        (core.transforms.TranslateX, [-0.5, 0.5], 0.8),
        (core.transforms.TranslateY, [-0.5, 0.5], 0.8),
        (core.transforms.Rotate, [-90, 90], 0.8),
        (core.transforms.AutoContrast, [0, 0], 0.8),
        (core.transforms.Invert, [0, 0], 0.8),
        (core.transforms.Equalize, [0, 0], 0.8),
        (core.transforms.HorizontalFlip, [0, 0], 0.8),
        (core.transforms.VerticalFlip, [0, 0], 0.8),
        (core.transforms.Solarize, [0, 256], 0.8),
        (core.transforms.Posterize, [0, 8], 0.8),
        (core.transforms.Contrast, [0.1, 1.9], 0.8),
        (core.transforms.Color, [0.1, 1.9], 0.8),
        (core.transforms.Brightness, [0.1, 1.9], 0.8),
        (core.transforms.Sharpness, [0.1, 1.9], 0.8),
        (core.transforms.Cutout, [0, 0.2], 0.8)
    ]

    # Custom dataset organizer
    # Instead of loading single data batch at one iteration,
    # Loads two data batches which would apply same data augmentation
    train_dataset = core.loader.AugConDatasetFolder(
        traindir,
        core.loader.AugConTransform(
            pre_process=transforms.Compose(pre_process),
            post_process=transforms.Compose(post_process),
            base_transforms=base_transforms))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint(args.save_dir, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, criterion, optimizer, epoch, args):
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
    # x1 = (data_batch1, augmented_data_batch1)
    # x2 = (data_batch2, augmented_data_batch2)
    # The applied augmentations for batch1 and batch2 are the same
    for i, (x1, x2, _, _, _, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            # Testing code for debugging data loader
            # transforms.functional.to_pil_image(x1[0][0]).save('./test01-0.png')
            # transforms.functional.to_pil_image(x1[1][0]).save('./test02-0.png')
            # transforms.functional.to_pil_image(x2[0][0]).save('./test11-0.png')
            # transforms.functional.to_pil_image(x2[1][0]).save('./test12-0.png')
            # transforms.functional.to_pil_image(x1[0][2]).save('./test01-1.png')
            # transforms.functional.to_pil_image(x1[1][2]).save('./test02-1.png')
            # transforms.functional.to_pil_image(x2[0][2]).save('./test11-1.png')
            # transforms.functional.to_pil_image(x2[1][2]).save('./test12-1.png')
            # assert(0)

            # Parsing data from batch1 (original/augmented)
            x1[0] = x1[0].cuda(args.gpu, non_blocking=True)
            x1[1] = x1[1].cuda(args.gpu, non_blocking=True)
            # Parsing data from batch2 (original/augmented)
            x2[0] = x2[0].cuda(args.gpu, non_blocking=True)
            x2[1] = x2[1].cuda(args.gpu, non_blocking=True)

        # compute output and loss
        # out = [similarity between positive pair, similarities between negative pairs . . .]
        # target = [1, 0, 0, 0, . . .]
        out1, target1, out2, target2 = model(im_x1_a1=x1[0], im_x1_a2=x1[1], im_x2_a1=x2[0], im_x2_a2=x2[1])
        # CrossEntropyLoss
        #print(out1.shape)
        #print(target1.shape)
        loss = criterion(out1, target1) + criterion(out2, target2)

        acc1, acc5 = accuracy(out1, target1, topk=(1, 5))
        losses.update(loss.item(), x1[0].size(0))
        top1.update(acc1[0], x1[0].size(0))
        top5.update(acc5[0], x1[0].size(0))
        #losses.update(loss.item(), x1[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
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


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

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

if __name__ == '__main__':
    main()