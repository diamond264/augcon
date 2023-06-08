#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import os
import random
import shutil
import time
import warnings
import math
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import core.loader
import core.builder
import core.transforms
import core.relnet
import core.resnet
import core.CPC

from data_loader.HHARDataset import HHARDataset

best_acc1 = 0

def run(config):
    cudnn.benchmark = True
    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if config.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if config.multiprocessing.dist_url == "env://" \
        and config.multiprocessing.world_size == -1:
        config.multiprocessing.world_size = int(os.environ["WORLD_SIZE"])

    config.multiprocessing.distributed = config.multiprocessing.world_size > 1 or \
        config.multiprocessing.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
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
    global best_acc1
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
        dist.init_process_group(
            backend=config.multiprocessing.dist_backend,
            init_method=config.multiprocessing.dist_url,
            world_size=config.multiprocessing.world_size,
            rank=config.multiprocessing.rank,
        )
    # create model
    print("=> creating model '{}'".format(config.model.type))
    
    if config.model.type == 'cpc':
        if config.train.meta_training:
            encoder = core.net.MetaCPC.Encoder(config.model.input_channels,
                                      config.model.z_dim)
            encoder = encoder.cuda(config.gpu)
            aggregator = core.net.MetaCPC.Aggregator(config.model.num_blocks,
                                                config.model.num_filters)
            aggregator = aggregator.cuda(config.gpu)
            model = core.net.MetaCPC.Classifier(encoder, aggregator,
                                            config.model.z_dim,
                                            config.model.seq_len,
                                            config.data.num_cls)
            if config.train.meta_training:
                da_model = core.net.MetaCPC.FuturePredictor(encoder, aggregator,
                                config.model.z_dim,
                                config.model.pred_steps,
                                config.model.n_negatives,
                                config.model.offset).cuda(config.gpu)
        else:
            encoder = core.net.CPC.Encoder(config.model.input_channels,
                                        config.model.z_dim)
            encoder = encoder.cuda(config.gpu)
            aggregator = core.net.CPC.Aggregator(config.model.num_blocks,
                                                config.model.num_filters)
            aggregator = aggregator.cuda(config.gpu)
            model = core.net.CPC.Classifier(encoder, aggregator,
                                            config.model.z_dim,
                                            config.model.seq_len,
                                            config.data.num_cls)

    # load from pre-trained, before DistributedDataParallel constructor
    if config.train.pretrained:
        if os.path.isfile(config.train.pretrained):
            print("=> loading checkpoint '{}'".format(config.train.pretrained))
            checkpoint = torch.load(config.train.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint["state_dict"]
            enc_idx = 0
            agg_idx = 0
            pred_idx = 0
            for k in list(state_dict.keys()):
                print(k)
                # retain only encoder_q up to before the embedding layer
                if config.model.type == 'cpc':
                    idx = 0
                    if config.train.meta_training:
                        if k.split('.')[idx] == 'encoder':
                            state_dict[f'encoder.vars.{enc_idx}'] = state_dict[k]
                            enc_idx += 1
                        elif k.split('.')[idx] == 'aggregator':
                            state_dict[f'aggregator.vars.{agg_idx}'] = state_dict[k]
                            agg_idx += 1
                        elif k.split('.')[idx] == 'predictor':
                            state_dict[f'vars.{pred_idx}'] = state_dict[k]
                            pred_idx += 1
                        pass
                    else:
                        if k.startswith("module.encoder") or k.startswith("module.aggregator"):
                            state_dict[k[len("module.") :]] = state_dict[k]
                        if k.split('.')[idx] == 'encoder':
                            state_dict[f'encoder.vars.{enc_idx}'] = state_dict[k]
                            enc_idx += 1
                        elif k.split('.')[idx] == 'aggregator':
                            state_dict[f'aggregator.vars.{agg_idx}'] = state_dict[k]
                            agg_idx += 1
                        elif k.split('.')[idx] == 'predictor':
                            state_dict[f'vars.{pred_idx}'] = state_dict[k]
                            pred_idx += 1
                        # delete renamed or unused k
                        del state_dict[k]
                    
            # print("model keys")
            # for k in list(model.state_dict().keys()):
            #     print(k)
            
            # print("encoder keys")
            # for k in list(encoder.state_dict().keys()):
            #     print(k)

            config.train.start_epoch = 0
            # msg = model.load_state_dict(state_dict, strict=False)
            # print("[loading weights] missing keys:")
            # print(msg.missing_keys)
            if config.train.meta_training:
                msg = da_model.load_state_dict(state_dict, strict=False)
            else:
                msg = model.load_state_dict(state_dict, strict=False)
            print("[loading weights] missing keys:")
            print(msg.missing_keys)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(config.train.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(config.train.pretrained))

    if config.multiprocessing.distributed:
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
            config.multiprocessing.workers = int((config.multiprocessing.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[config.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
    else:
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
        pass

    # optionally resume from a checkpoint
    if config.train.resume:
        if os.path.isfile(config.train.resume):
            print("=> loading checkpoint '{}'".format(config.train.resume))
            if config.gpu is None:
                checkpoint = torch.load(config.train.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(config.gpu)
                checkpoint = torch.load(config.train.resume, map_location=loc)
            config.train.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if config.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(config.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    config.train.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(config.train.resume))

    if config.data.name == 'hhar':
        with open(config.data.train_dataset_path, 'rb') as f:
            train_dataset = pickle.load(f)
        with open(config.data.test_dataset_path, 'rb') as f:
            test_dataset = pickle.load(f)
        with open(config.data.val_dataset_path, 'rb') as f:
            val_dataset = pickle.load(f)
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
            fixed_data_path=config.data.fixed_data_path,
            complementary=False
        )
        dataset.filter_domain(user=config.data.user, model=config.data.model)
    
    # train_size = config.data.shot_num*config.data.num_cls
    # test_size = config.data.test_size
    # val_size = config.data.val_size
    # print(f'train size: {train_size}, test size: {test_size}, val size: {val_size}')
    # train_dataset, test_dataset, val_dataset = dataset.split_kshot_dataset(shot_num=config.data.shot_num, test_size=test_size, val_size=val_size)
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    if config.multiprocessing.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.multiprocessing.workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.multiprocessing.workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.multiprocessing.workers,
        pin_memory=True,
    )
    
    
    # domain adaptation
    if config.train.perform_da:
        da_criterion = nn.CrossEntropyLoss().cuda(config.gpu)
        neg_support_set = torch.load('neg_support_set.pth')
        domain_adaptation(train_loader, neg_support_set, da_model, da_criterion, config)
        # assert(0)
    
    # freeze all layers but the last fc
    if config.train.freeze:
        for name, param in model.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
    # # init the fc layer
    # model.fc.weight.data.normal_(mean=0.0, std=0.01)
    # model.fc.bias.data.zero_()
    
    # define loss function (criterion) and optimizer
    if config.train.criterion == 'crossentropy':
        criterion = nn.CrossEntropyLoss().cuda(config.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(f'parameters except {len(parameters)} are frozen...')
    # assert len(parameters) == 2  # fc.weight, fc.bias
    
    if config.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            parameters, config.train.lr, momentum=config.train.momentum,
            weight_decay=config.train.weight_decay
        )
    elif config.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            parameters, config.train.lr, 
            weight_decay=config.train.weight_decay
        )
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
                loc = "cuda:{}".format(config.gpu)
                checkpoint = torch.load(config.train.resume, map_location=loc)
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("=> no checkpoint found at '{}'".format(config.train.resume))
    
    if config.evaluate:
        validate(test_loader, model, criterion, config)
        return

    for epoch in range(config.train.start_epoch, config.train.epochs):
        if config.multiprocessing.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, config)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, config)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, config)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not config.multiprocessing.multiprocessing_distributed or (
            config.multiprocessing.multiprocessing_distributed and config.multiprocessing.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                config.train.save_dir,
                {
                    "epoch": epoch + 1,
                    "arch": config.model.type,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
            )
            # if epoch == config.train.start_epoch:
            #     sanity_check(model.state_dict(), config.train.pretrained)
    validate(test_loader, model, criterion, config)


def domain_adaptation(train_loader, neg_support_set, da_model, da_criterion, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        1,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(0))
    
    da_model.train()
    da_model.zero_grad()
        
    da_dataset = train_loader.dataset.tensors[0].cuda(config.gpu)#[:10]
    
    idx = 0
    end = time.time()
    fast_weights = da_model.parameters()
    for step in range(config.train.adaptation_steps):
        # logits, targets, _ = da_model(da_dataset, neg_support_set, fast_weights)
        logits, targets, _ = da_model(da_dataset, da_dataset, fast_weights)
        loss = da_criterion(logits, targets)
        grad = torch.autograd.grad(loss, fast_weights)
        fast_weights = [w-config.train.task_lr*grad[i] for i, w in enumerate(fast_weights)]
        # grad = torch.autograd.grad(loss, fast_weights[-2:])
        # fast_weights = list(fast_weights[:-2])+[w-config.train.task_lr*grad[i] for i, w in enumerate(fast_weights[-2:])]
        # grad = torch.autograd.grad(loss, fast_weights)
        # fast_weights = [w-config.train.task_lr*grad[i] for i, w in enumerate(fast_weights)]
        
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        losses.update(loss.item(), logits.size(0))
        top1.update(acc1[0], logits.size(0))
        top5.update(acc5[0], logits.size(0))
        losses.update(loss.item(), logits.size(0))
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        progress.display(idx)
        idx += 1
        
    da_model.encoder.vars = nn.ParameterList(fast_weights[:da_model.enc_param_idx])
    da_model.aggregator.vars = nn.ParameterList(fast_weights[da_model.enc_param_idx:da_model.agg_param_idx])
    # assert(0)


def train(train_loader, model, criterion, optimizer, epoch, config):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    if config.train.freeze:
        model.eval()

    end = time.time()
    for i, x in enumerate(train_loader):
        feature = x[0].cuda(config.gpu)
        class_label = x[1].cuda(config.gpu)
        domain_label = x[2].cuda(config.gpu)

        # compute output
        output = model(feature)
        loss = criterion(output, class_label)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, class_label, topk=(1, 5))
        losses.update(loss.item(), feature.size(0))
        top1.update(acc1[0], feature.size(0))
        top5.update(acc5[0], feature.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.train.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, config):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, x in enumerate(val_loader):
            feature = x[0].cuda(config.gpu)
            class_label = x[1].cuda(config.gpu)
            domain_label = x[2].cuda(config.gpu)

            # compute output
            output = model(feature)
            loss = criterion(output, class_label)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, class_label, topk=(1, 5))
            losses.update(loss.item(), feature.size(0))
            top1.update(acc1[0], feature.size(0))
            top5.update(acc5[0], feature.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.test.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg


def save_checkpoint(save_dir, state, is_best, filename="checkpoint.pth.tar"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filepath = os.path.join(save_dir, filename)
    best_path = os.path.join(save_dir, "model_best.pth.tar")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, best_path)


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint["state_dict"]

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if "fc.weight" in k or "fc.bias" in k:
            continue

        # name in pretrained model
        k_pre = (
            "module.encoder_q." + k[len("module.") :]
            if k.startswith("module.")
            else "module.encoder_q." + k
        )

        assert (
            state_dict[k].cpu() == state_dict_pre[k_pre]
        ).all(), "{} is changed in linear classifier training.".format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


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
    