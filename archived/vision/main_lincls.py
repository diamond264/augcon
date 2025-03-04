#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

import core.utils
import core.loader
import core.builder
import core.transforms
import core.relnet
import core.resnet

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    default='/mnt/sting/hjyoon/projects/cross/ImageNet_ILSVRC2012_mini',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-bval', '--batch-size-val', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument('--lars', action='store_true',
                    help='Use LARS')
parser.add_argument('--checkpoint', default='', type=str,
                    help='path to fineture checkpoint')

#additional config for fineturing option
parser.add_argument('--tune', default='ed', type=str,
                    help='finetune option ed for freeze encoder and discriminator, edf for freeze only encoder and e for using encoder only')

### AUGCONTRAST$ SPECIFIC CONFIGS ###
parser.add_argument('--temp', default=0.07, type=float,
                    help='softmax temperatqure parameter (default: 0.07)')

best_acc1 = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
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
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'resnet18':
        encoder = core.resnet.Encoder_res18()
        discriminator = core.resnet.Discriminator_res() 
        model = core.builder.AugCon_eval(
            encoder, discriminator, mode= args.tune)

    elif args.arch == 'relnet':
        encoder = core.relnet.CNNEncoder()
        discriminator = core.relnet.RelationNetwork()
        model = core.builder.AugCon_eval(encoder, discriminator, mode=args.tune)

    # model.fc.weight.data.normal_(mean=0.0, std=0.01)
    # model.fc.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    # model.simclr_head = nn.Linear(64, 1)
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            # print(state_dict.keys())
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.') and not k.startswith('module.fc'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            args.start_epoch = 0
            # for k in list(state_dict.keys()):
            #     if k in list(model.state_dict().keys()): pass
            #     else: print(k)
            
            # for k in list(model.state_dict().keys()):
            #     if k in list(state_dict.keys()): pass
            #     else: print(k)
            # assert(0)
            msg = model.load_state_dict(state_dict, strict=True)

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
    
    model.simclr_head = nn.Linear(64, 1)

    if args.tune == 'ed' or args.tune == 'e':
        for name, param in model.named_parameters():
            # if name not in ['simclr_head.weight', 'simclr_head.bias']:
            print(name)
            if name.startswith('encoder') or name.startswith('discriminator'):
            # if name not in ['fc.weight', 'fc.bias', 'fc1', 'act']:
                param.requires_grad = False
                # param.requires_grad = True
    # elif args.tune == 'edf' :
    #     for name, param in model.named_parameters():
    #         print(name)
    #         if name.startswith('encoder'):
    #             param.requires_grad = False

    # sanity_check(model.state_dict(), args.pretrained)
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(len(parameters))

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256
    init_lr = args.lr

    if args.distributed:
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
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, broadcast_buffers=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)#, broadcast_buffers=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # assert len(parameters) == 2  # fc.weight, fc.bias

    # optimizer = torch.optim.SGD(parameters, init_lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(parameters, init_lr,weight_decay=args.weight_decay)

    if args.lars:
        print("=> use LARS optimizer.")
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)

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
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True
    # print(model)
    # assert(0)
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform= transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])

    #train_dataset = core.loader.Train_cls_loader(traindir, transform) 
    ref_sample = None
    if args.tune == 'ed' or args.tune =='edf':
        train_dataset = core.loader.Val_cls_loader(traindir, transform)
        val_dataset = core.loader.Val_cls_loader(valdir, transform)
        ref_dataset = core.loader.Ref_loader(traindir, transform)
        ref_sample= torch.utils.data.DataLoader(
            ref_dataset, 1, shuffle=False, pin_memory=True)
    else:
        train_dataset = core.loader.Train_cls_loader2(traindir, transform)
        val_dataset = core.loader.Train_cls_loader2(valdir, transform)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1024, shuffle= False,
        num_workers=args.workers, pin_memory=True)

    

    #print(ref_sample.shape)
    '''
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    '''
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        #train(train_loader, model, criterion, optimizer, epoch, args)
        train2(train_loader, ref_sample, model, criterion, optimizer, epoch, args)
        # evaluate on validation set
        acc1 = validate(val_loader, ref_sample, model, criterion, args)

        # remember best acc@1 and save checkpoint
        
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args)
            
            # if epoch == args.start_epoch:
            #     sanity_check(model.state_dict(), args.pretrained)
            
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    torch.autograd.set_detect_anomaly(True)
    for i, (sample, positive, pos_lab, negative, neg_lab, target, negative_target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            sample = sample.cuda(args.gpu, non_blocking=True)
            positive = positive.cuda(args.gpu, non_blocking=True)
            pos_lab = pos_lab.cuda(args.gpu, non_blocking=True)
            negative = negative.cuda(args.gpu, non_blocking=True)
            neg_lab = neg_lab.cuda(args.gpu, non_blocking=True)

        # compute output
        #print(model(sample,positive).shape)
        #print(pos_lab.shape)
        out_pos= model(sample,positive )
        #out_pos.register_hook(lambda grad: print('out_pos',grad.sum()))

        out_neg= model(sample,negative)
        #out_neg.register_hook(lambda grad: print('out_neg', grad.sum()))
        #print(out_pos.shape)
        #print(pos_lab.shape)
        out = torch.cat((out_pos,out_neg), 1)
        #pred = torch.cat((pos_lab,neg_lab) ,1)
        loss = criterion(out, neg_lab)

        # measure accuracy and record loss
        acc1 = accuracy(out, neg_lab, topk=(1,))
        losses.update(loss.item(), sample.size(0))
        top1.update(acc1[0].item(), sample.size(0))
        #top5.update(acc5[0], images.size(0))
        #print(top1)
        # compute gradient and do SGD step
        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


# Negative sample per class
def train2(train_loader, ref_sample, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()
    # model.train()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        if args.tune == 'ed' or args.tune == 'edf':
            ref_samplei = next(ref_sample.__iter__())
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        if args.tune == 'ed' or args.tune == 'edf':
            ref_samplei =ref_samplei.cuda(args.gpu, non_blocking=True)
            batch_size, c, h, w = images.shape
            _, num_class, c1, h1, w1 = ref_samplei.shape
            images_ext = images.unsqueeze(0).repeat(num_class,1,1,1,1).transpose(0,1).reshape(-1,c,h,w)     
            ref_sample_ext = ref_samplei.repeat(batch_size,1,1,1,1).reshape(-1,c1,h1,w1)
            # compute output
            output = model(images_ext, ref_sample_ext).reshape(batch_size, num_class)
        else:
            output= model(images, None)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, = accuracy(output, target, topk=(1, ))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % (args.print_freq) == 0:
            progress.display(i)


def validate(val_loader, ref_sample, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.tune =='ed' or args.tune == 'edf':
                ref_samplei = next(ref_sample.__iter__())
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            if args.tune =='ed' or args.tune == 'edf':
                ref_samplei = ref_samplei.cuda(args.gpu, non_blocking=True)
                batch_size, c, h, w = images.shape
                _, num_class, c1, h1, w1 = ref_samplei.shape
                images_ext = images.unsqueeze(0).repeat(num_class,1,1,1,1).transpose(0,1).reshape(-1,c,h,w)     
                ref_sample_ext = ref_samplei.repeat(batch_size,1,1,1,1).reshape(-1,c1,h1,w1)
                # compute output
                output = model(images_ext, ref_sample_ext).reshape(batch_size, num_class)
            else:
                output = model(images, None)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1, ))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % (args.print_freq*1000) == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} '
              .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    torch.save(state,args.checkpoint+filename)
    if is_best:
        shutil.copyfile(args.checkpoint+filename, args.checkpoint+'model_best.pth.tar')


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = k \
            if k.startswith('module.') else 'module.' + k
        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


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
    print(cur_lr)
    for param_group in optimizer.param_groups:
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