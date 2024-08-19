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

from torch.utils.data import DataLoader, Dataset, DistributedSampler


from torch.utils.tensorboard import SummaryWriter


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
            w, b = vars[idx], vars[idx + 1]
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


class TaskspecificHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_cls):
        super(TaskspecificHead, self).__init__()
        self.vars = nn.ParameterList()

        fc1 = nn.Linear(input_size, hidden_size)
        fc2 = nn.Linear(hidden_size, num_cls)

        w = fc1.weight
        b = fc1.bias
        self.vars.append(w)
        self.vars.append(b)
        w = fc2.weight
        b = fc2.bias
        self.vars.append(w)
        self.vars.append(b)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars

        x = F.linear(x, vars[0], vars[1])
        x = F.relu(x, True)
        x = F.linear(x, vars[2], vars[3])
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


class TPNNet(nn.Module):
    def __init__(self, input_channels=3, z_dim=96, out_dim=2):
        super(TPNNet, self).__init__()
        self.encoder = Encoder(input_channels, z_dim)

        self.noise_head = TaskspecificHead(z_dim, 256, out_dim)
        self.scale_head = TaskspecificHead(z_dim, 256, out_dim)
        self.rotate_head = TaskspecificHead(z_dim, 256, out_dim)
        self.negate_head = TaskspecificHead(z_dim, 256, out_dim)
        self.flip_head = TaskspecificHead(z_dim, 256, out_dim)
        self.permute_head = TaskspecificHead(z_dim, 256, out_dim)
        self.time_warp_head = TaskspecificHead(z_dim, 256, out_dim)
        self.channel_shuffle_head = TaskspecificHead(z_dim, 256, out_dim)

    def forward(self, x, vars=None):
        if vars is None:
            vars = nn.ParameterList()
            vars.extend(self.encoder.parameters())

            vars.extend(self.noise_head.parameters())
            vars.extend(self.scale_head.parameters())
            vars.extend(self.rotate_head.parameters())
            vars.extend(self.negate_head.parameters())
            vars.extend(self.flip_head.parameters())
            vars.extend(self.permute_head.parameters())
            vars.extend(self.time_warp_head.parameters())
            vars.extend(self.channel_shuffle_head.parameters())

        enc_vars = vars[: len(self.encoder.parameters())]
        noise_vars = vars[
            len(self.encoder.parameters()) : len(self.encoder.parameters()) + 4
        ]
        scale_vars = vars[
            len(self.encoder.parameters()) + 4 : len(self.encoder.parameters()) + 8
        ]
        rotate_vars = vars[
            len(self.encoder.parameters()) + 8 : len(self.encoder.parameters()) + 12
        ]
        negate_vars = vars[
            len(self.encoder.parameters()) + 12 : len(self.encoder.parameters()) + 16
        ]
        flip_vars = vars[
            len(self.encoder.parameters()) + 16 : len(self.encoder.parameters()) + 20
        ]
        permute_vars = vars[
            len(self.encoder.parameters()) + 20 : len(self.encoder.parameters()) + 24
        ]
        time_vars = vars[
            len(self.encoder.parameters()) + 24 : len(self.encoder.parameters()) + 28
        ]
        channel_vars = vars[len(self.encoder.parameters()) + 28 :]

        z = self.encoder(x, enc_vars)

        noise_y = self.noise_head(z, noise_vars)
        scale_y = self.scale_head(z, scale_vars)
        rotate_y = self.rotate_head(z, rotate_vars)
        negate_y = self.negate_head(z, negate_vars)
        flip_y = self.flip_head(z, flip_vars)
        permute_y = self.permute_head(z, permute_vars)
        time_warp_y = self.time_warp_head(z, time_vars)
        channel_shuffle_y = self.channel_shuffle_head(z, channel_vars)

        return (
            noise_y,
            scale_y,
            rotate_y,
            negate_y,
            flip_y,
            permute_y,
            time_warp_y,
            channel_shuffle_y,
        )

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                self.encoder.zero_grad()
                self.noise_head.zero_grad()
                self.scale_head.zero_grad()
                self.rotate_head.zero_grad()
                self.negate_head.zero_grad()
                self.flip_head.zero_grad()
                self.permute_head.zero_grad()
                self.time_warp_head.zero_grad()
                self.channel_shuffle_head.zero_grad()

            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        vars = nn.ParameterList()
        vars.extend(self.encoder.parameters())
        vars.extend(self.noise_head.parameters())
        vars.extend(self.scale_head.parameters())
        vars.extend(self.rotate_head.parameters())
        vars.extend(self.negate_head.parameters())
        vars.extend(self.flip_head.parameters())
        vars.extend(self.permute_head.parameters())
        vars.extend(self.time_warp_head.parameters())
        vars.extend(self.channel_shuffle_head.parameters())

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


class TPNClassifier(nn.Module):
    def __init__(self, input_channels, z_dim, num_cls, mlp=True):
        super(TPNClassifier, self).__init__()
        self.base_model = Encoder(input_channels, z_dim)
        self.classifier = ClassificationHead(z_dim, z_dim, num_cls, mlp)

    def forward(self, x):
        x = self.base_model(x)
        pred = self.classifier(x)
        return pred


class MetaTPNLearner:
    def __init__(self, cfg, gpu, logger):
        self.cfg = cfg
        self.gpu = gpu
        self.logger = logger

    def run(self, train_dataset, val_dataset, test_dataset):
        num_gpus = len(self.gpu)
        logs = mp.Manager().list([])
        self.logger.info("Executing Meta TPN")
        self.logger.info("Logs are skipped during training")
        if num_gpus > 1:
            mp.spawn(
                self.main_worker,
                args=(num_gpus, train_dataset, val_dataset, test_dataset, logs),
                nprocs=num_gpus,
            )
        else:
            self.main_worker(0, 1, train_dataset, val_dataset, test_dataset, logs)

        for log in logs:
            self.logger.info(log)

    def main_worker(
        self, rank, world_size, train_dataset, val_dataset, test_dataset, logs
    ):
        # Model initialization
        net = TPNNet(self.cfg.input_channels, self.cfg.z_dim, self.cfg.out_dim)
        if self.cfg.mode == "finetune" or self.cfg.mode == "eval":
            cls_net = TPNClassifier(
                self.cfg.input_channels, self.cfg.z_dim, self.cfg.num_cls, self.cfg.mlp
            )

        # DDP setting
        if world_size > 1:
            dist.init_process_group(
                backend="nccl",
                init_method=self.cfg.dist_url,
                world_size=world_size,
                rank=rank,
            )
            torch.cuda.set_device(rank)
            net.cuda()
            net = nn.parallel.DistributedDataParallel(
                net, device_ids=[rank], find_unused_parameters=True
            )
            if self.cfg.mode == "finetune" or self.cfg.mode == "eval":
                cls_net.cuda()
                cls_net = nn.parallel.DistributedDataParallel(
                    cls_net, device_ids=[rank], find_unused_parameters=True
                )

            train_sampler = DistributedSampler(train_dataset)
            if self.cfg.mode == "finetune" or self.cfg.mode == "eval":
                if len(val_dataset) > 0:
                    val_sampler = DistributedSampler(val_dataset)
                test_sampler = DistributedSampler(test_dataset)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.cfg.batch_size // world_size,
                    shuffle=False,
                    sampler=train_sampler,
                    num_workers=self.cfg.num_workers,
                    drop_last=True,
                )
                if len(val_dataset) > 0:
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=self.cfg.batch_size // world_size,
                        shuffle=False,
                        sampler=val_sampler,
                        num_workers=self.cfg.num_workers,
                        drop_last=True,
                    )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.cfg.batch_size // world_size,
                    shuffle=False,
                    sampler=test_sampler,
                    num_workers=self.cfg.num_workers,
                    drop_last=True,
                )
            meta_train_dataset = torch.utils.data.Subset(
                train_dataset, list(train_sampler)
            )
            if rank == 0:
                log = "DDP is used for training - training {} instances for each worker".format(
                    len(list(train_sampler))
                )
                logs.append(log)
                print(log)
        else:
            torch.cuda.set_device(rank)
            net.cuda()
            if self.cfg.mode == "finetune" or self.cfg.mode == "eval":
                cls_net.cuda()
            if self.cfg.mode == "finetune" or self.cfg.mode == "eval":
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    num_workers=self.cfg.num_workers,
                    drop_last=False,
                )
                if len(val_dataset) > 0:
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=self.cfg.batch_size,
                        shuffle=True,
                        num_workers=self.cfg.num_workers,
                        drop_last=False,
                    )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    num_workers=self.cfg.num_workers,
                    drop_last=False,
                )
            meta_train_dataset = train_dataset
            if rank == 0:
                log = "Single GPU is used for training - training {} instances for each worker".format(
                    len(train_dataset)
                )
                logs.append(log)
                print(log)

        # Define criterion
        if self.cfg.criterion == "crossentropy":
            criterion = nn.CrossEntropyLoss().cuda()

        if self.cfg.mode == "pretrain":
            parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
            if self.cfg.optimizer == "sgd":
                optimizer = torch.optim.SGD(
                    parameters,
                    self.cfg.lr,
                    momentum=self.cfg.momentum,
                    weight_decay=self.cfg.wd,
                )
            elif self.cfg.optimizer == "adam":
                optimizer = torch.optim.Adam(
                    parameters, self.cfg.lr, weight_decay=self.cfg.wd
                )

            # Load checkpoint if exists
            if os.path.isfile(self.cfg.resume):
                if rank == 0:
                    log = "Loading state_dict from checkpoint - {}".format(
                        self.cfg.resume
                    )
                    logs.append(log)
                    print(log)
                loc = "cuda:{}".format(rank)
                state = torch.load(self.cfg.resume, map_location=loc)
                for k, v in list(state["state_dict"].items()):
                    if world_size > 1:
                        k = "module." + k
                    if k in net.state_dict().keys():
                        state["state_dict"][k] = v

                net.load_state_dict(state["state_dict"], strict=False)
                optimizer.load_state_dict(state["optimizer"])
                self.cfg.start_epoch = state["epoch"]

            # Pretrain
            indices_per_domain = self.split_per_domain(meta_train_dataset)

            for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
                if world_size > 1:
                    train_sampler.set_epoch(epoch)

                supports = []
                queries = []
                target_supports = []
                target_queries = []
                if self.cfg.task_per_domain:
                    supports, queries, target_supports, target_queries = (
                        self.gen_per_domain_tasks(
                            meta_train_dataset,
                            indices_per_domain,
                            self.cfg.task_size,
                            self.cfg.num_task,
                        )
                    )
                if self.cfg.multi_cond_num_task > 0:
                    (
                        multi_cond_supports,
                        multi_cond_queries,
                        target_multi_cond_supports,
                        target_multi_cond_queries,
                    ) = self.gen_random_tasks(
                        meta_train_dataset,
                        self.cfg.task_size,
                        self.cfg.multi_cond_num_task,
                    )
                    supports = supports + multi_cond_supports
                    queries = queries + multi_cond_queries
                    target_supports = target_supports + target_multi_cond_supports
                    target_queries = target_queries + target_multi_cond_queries

                self.pretrain(
                    rank,
                    net,
                    supports,
                    queries,
                    target_supports,
                    target_queries,
                    criterion,
                    optimizer,
                    epoch,
                    self.cfg.epochs,
                    logs,
                )

                if rank == 0 and (epoch + 1) % self.cfg.save_freq == 0:
                    ckpt_dir = self.cfg.ckpt_dir
                    ckpt_filename = "checkpoint_{:04d}.pth.tar".format(epoch)
                    ckpt_filename = os.path.join(ckpt_dir, ckpt_filename)
                    if world_size > 1:
                        state_dict = net.module.state_dict()
                    else:
                        state_dict = net.state_dict()
                    self.save_checkpoint(ckpt_filename, epoch, state_dict, optimizer)

        elif self.cfg.mode == "finetune" or self.cfg.mode == "eval":
            # Freeze the encoder part of the network
            if self.cfg.freeze:
                for name, param in cls_net.named_parameters():
                    if not "classifier" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            # Defining optimizer for classifier
            parameters = list(filter(lambda p: p.requires_grad, cls_net.parameters()))
            if self.cfg.optimizer == "sgd":
                optimizer = torch.optim.SGD(
                    parameters,
                    self.cfg.lr,
                    momentum=self.cfg.momentum,
                    weight_decay=self.cfg.wd,
                )
            elif self.cfg.optimizer == "adam":
                optimizer = torch.optim.Adam(
                    parameters, self.cfg.lr, weight_decay=self.cfg.wd
                )

            if not os.path.isfile(self.cfg.resume):
                # Load pretrained model to net
                if os.path.isfile(self.cfg.pretrained):
                    if rank == 0:
                        log = "Loading pretrained model from checkpoint - {}".format(
                            self.cfg.pretrained
                        )
                        logs.append(log)
                        print(log)
                    loc = "cuda:{}".format(rank)
                    state = torch.load(self.cfg.pretrained, map_location=loc)[
                        "state_dict"
                    ]

                    if self.cfg.no_vars:
                        enc_dict = {}
                        for idx, k in enumerate(list(net.state_dict().keys())):
                            if not "classifier" in k:
                                enc_dict[k] = list(state.items())[idx][1]
                        msg = net.load_state_dict(enc_dict, strict=False)
                    else:
                        for k, v in list(state.items()):
                            if world_size > 1:
                                k = "module." + k
                            if k in net.state_dict().keys():
                                state[k] = v

                        msg = net.load_state_dict(state, strict=False)
                    if rank == 0:
                        log = "Missing keys: {}".format(msg.missing_keys)
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
                    net.eval()
                    net.zero_grad()
                    support = []
                    target_support = []
                    for e in meta_train_dataset:
                        support.append(e[1])
                        target_support.append(torch.tensor(e[2]))
                    support = torch.stack(support, dim=0).cuda()
                    target_support = torch.stack(target_support, dim=0).cuda()
                    enc_parameters = self.meta_train(
                        rank,
                        net,
                        support,
                        target_support,
                        criterion,
                        log_internals=True,
                        logs=logs,
                    )
                else:
                    enc_parameters = list(net.parameters())

                if rank == 0:
                    log = "Loading encoder parameters to the classifier"
                    logs.append(log)
                    print(log)

                enc_dict = {}
                for idx, k in enumerate(list(cls_net.state_dict().keys())):
                    if not "classifier" in k:
                        enc_dict[k] = enc_parameters[idx]

                msg = cls_net.load_state_dict(enc_dict, strict=False)
                if rank == 0:
                    log = "Missing keys: {}".format(msg.missing_keys)
                    logs.append(log)
                    print(log)
            else:
                if rank == 0:
                    log = "Loading state_dict from checkpoint - {}".format(
                        self.cfg.resume
                    )
                    logs.append(log)
                    print(log)
                loc = "cuda:{}".format(rank)
                state = torch.load(self.cfg.resume, map_location=loc)
                for k, v in list(state["state_dict"].items()):
                    if world_size > 1:
                        k = "module." + k
                    if k in cls_net.state_dict().keys():
                        state["state_dict"][k] = v

                cls_net.load_state_dict(state["state_dict"], strict=False)
                optimizer.load_state_dict(state["optimizer"])
                self.cfg.start_epoch = state["epoch"]

            if self.cfg.mode == "finetune":
                for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
                    if world_size > 1:
                        train_sampler.set_epoch(epoch)
                        if len(val_dataset) > 0:
                            val_sampler.set_epoch(epoch)
                        test_sampler.set_epoch(epoch)

                    self.finetune(
                        rank,
                        cls_net,
                        train_loader,
                        criterion,
                        optimizer,
                        epoch,
                        self.cfg.epochs,
                        logs,
                    )
                    # if len(val_dataset) > 0:
                    #     self.validate(rank, cls_net, val_loader, criterion, logs)

                    if rank == 0 and (epoch + 1) % self.cfg.save_freq == 0:
                        ckpt_dir = self.cfg.ckpt_dir
                        ckpt_filename = "checkpoint_{:04d}.pth.tar".format(epoch)
                        ckpt_filename = os.path.join(ckpt_dir, ckpt_filename)
                        state_dict = cls_net.state_dict()
                        if world_size > 1:
                            for k, v in list(state_dict.items()):
                                if "module." in k:
                                    state_dict[k.replace("module.", "")] = v
                                    del state_dict[k]
                        self.save_checkpoint(
                            ckpt_filename, epoch, state_dict, optimizer
                        )

            self.validate_finetune(rank, cls_net, test_loader, criterion, logs)

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

    def gen_per_domain_tasks(
        self, dataset, indices_per_domain, task_size, num_task=None
    ):
        supports = []
        queries = []
        target_supports = []
        target_queries = []

        with torch.no_grad():
            if num_task is None:
                for _, indices in indices_per_domain.items():
                    random.shuffle(indices)
                    suppor_ = torch.utils.data.Subset(dataset, indices[:task_size])
                    target_support = []
                    support = []
                    for data in support_:
                        target_support.append(torch.tensor(data[2]))
                        support.append(data[1])
                    support = torch.stack(support, dim=0)
                    target_support = torch.stack(target_support, dim=0)

                    query_ = torch.utils.data.Subset(
                        dataset, indices[task_size : 2 * task_size]
                    )
                    target_query = []
                    query = []
                    for data in query_:
                        target_query.append(torch.tensor(data[2]))
                        query.append(data[1])
                    print(query)
                    query = torch.stack(query, dim=0)
                    target_query = torch.stack(target_query, dim=0)
                    supports.append(support)
                    queries.append(query)
                    target_supports.append(target_support)
                    target_queries.append(target_query)
            else:
                for _ in range(num_task):
                    dom = random.choice(list(indices_per_domain.keys()))
                    indices = indices_per_domain[dom]
                    random.shuffle(indices)
                    support_ = torch.utils.data.Subset(dataset, indices[:task_size])
                    support = []
                    target_support = []

                    for e in support_:
                        support.append(e[1])
                        target_support.append(torch.tensor(e[2]))
                    if len(support) >= task_size:
                        support = support[:task_size]
                        target_support = target_support[:task_size]
                    else:
                        support = (
                            support * (task_size // len(support))
                            + support[: task_size % len(support)]
                        )
                        target_support = (
                            target_support * (task_size // len(target_support))
                            + target_support[: task_size % len(target_support)]
                        )
                        # supplementary = random.sample(support, task_size % len(support))
                        # selected_indices = [support.index(element) for element in supplementary]
                        # support = support * (task_size // len(support)) + supplementary
                        # target_support = target_support * (task_size // len(target_support)) + [target_support[index] for index in selected_indices]

                    support = torch.stack(support, dim=0)
                    target_support = torch.stack(target_support, dim=0)
                    # print(target_support)
                    # print(target_support.shape)

                    # query = torch.utils.data.Subset(dataset, indices[task_size:2*task_size])
                    query_ = torch.utils.data.Subset(
                        dataset, indices[task_size : 2 * task_size]
                    )
                    query = []
                    target_query = []
                    for e in query_:
                        query.append(e[1])
                        target_query.append(torch.tensor(e[2]))
                    if len(query) >= task_size:
                        query = query[:task_size]
                        target_query = target_query[:task_size]
                    else:
                        query = (
                            query * (task_size // len(query))
                            + query[: task_size % len(query)]
                        )
                        target_query = (
                            target_query * (task_size // len(target_query))
                            + target_query[: task_size % len(target_query)]
                        )
                        # supplementary = random.sample(query, task_size % len(query))
                        # selected_indices = [query.index(element) for element in supplementary]
                        # query = query * (task_size // len(query)) + supplementary
                        # target_query = target_query * (task_size // len(target_query)) + [target_query[index] for index in selected_indices]

                    query = torch.stack(query, dim=0)
                    target_query = torch.stack(target_query, dim=0)
                    supports.append(support)
                    queries.append(query)
                    target_supports.append(target_support)
                    target_queries.append(target_query)

        return supports, queries, target_supports, target_queries

    def gen_random_tasks(self, dataset, task_size, num_task):
        supports = []
        queries = []
        target_supports = []
        target_queries = []
        with torch.no_grad():
            indices = list(range(len(dataset)))
            random.shuffle(indices)

            st = 0
            ed = task_size
            for _ in range(num_task):
                support_ = torch.utils.data.Subset(dataset, indices[st:ed])
                target_support = []
                support = []
                for data in support_:
                    target_support.append(torch.tensor(data[2]))
                    support.append(data[1])
                support = torch.stack(support, dim=0)
                target_support = torch.stack(target_support, dim=0)
                st += task_size
                ed += task_size

                query_ = torch.utils.data.Subset(dataset, indices[st:ed])
                target_query = []
                query = []
                for data in query_:
                    target_query.append(torch.tensor(data[2]))
                    query.append(data[1])
                query = torch.stack(query, dim=0)
                target_query = torch.stack(target_query, dim=0)
                st += task_size
                ed += task_size
                supports.append(support)
                queries.append(query)
                target_supports.append(target_support)
                target_queries.append(target_query)

        return supports, queries, target_supports, target_queries

    def pretrain(
        self,
        rank,
        net,
        supports,
        queries,
        target_supports,
        target_queries,
        criterion,
        optimizer,
        epoch,
        num_epochs,
        logs,
    ):
        net.train()
        net.zero_grad()

        log = f"Epoch [{epoch + 1}/{num_epochs}]"
        if rank == 0:
            logs.append(log)
            print(log)

        q_losses = []
        for task_idx in range(len(supports)):
            support = supports[task_idx].cuda().float()
            query = queries[task_idx].cuda()
            target_support = target_supports[task_idx].cuda().long()
            target_query = target_queries[task_idx].cuda()

            fast_weights = self.meta_train(
                rank,
                net,
                support,
                target_support,
                criterion,
                log_internals=self.cfg.log_meta_train,
                logs=logs,
            )

            q_logits = net(query, fast_weights)
            q_loss = 0
            for i in range(len(q_logits)):
                q_loss += criterion(q_logits[i], target_query[:, i])
            q_loss /= len(q_logits)
            q_losses.append(q_loss)

            if task_idx % self.cfg.log_freq == 0:
                q_logits = torch.cat(q_logits)
                target_query = torch.transpose(target_query, 0, 1)
                target_query = target_query.reshape(
                    -1,
                )
                acc1, acc3 = self.accuracy(q_logits, target_query, topk=(1, 1))

                log = f"\t({task_idx}/{len(supports)}) "
                log += f"Loss: {q_loss.item():.4f}, Acc(1): {acc1.item():.2f}"
                if rank == 0:
                    logs.append(log)
                    print(log)

        q_losses = torch.stack(q_losses, dim=0)
        loss = torch.sum(q_losses)
        # reg_term = torch.sum((q_losses - torch.mean(q_losses))**2)
        # loss += reg_term * self.cfg.reg_lambda
        loss = loss / len(supports)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def meta_train(
        self,
        rank,
        net,
        support,
        target_support,
        criterion,
        log_internals=False,
        logs=None,
    ):
        fast_weights = list(net.parameters())
        for i in range(self.cfg.task_steps):
            shuffled_idx = torch.randperm(len(support))
            support = support[shuffled_idx]
            target_support = target_support[shuffled_idx]

            s_logits = net(support, fast_weights)
            s_loss = 0
            # print(support.shape, target_support.shape, s_logits[0].shape)
            for j in range(len(s_logits)):
                # print(s_logits[i].shape)
                # print(target_support[:,i].shape)
                s_loss += criterion(s_logits[j], target_support[:, j])
            s_loss /= len(s_logits)

            grad = torch.autograd.grad(s_loss, fast_weights)
            fast_weights = list(
                map(lambda p: p[1] - self.cfg.task_lr * p[0], zip(grad, fast_weights))
            )

            if log_internals and rank == 0:
                TPN_pred = torch.cat(s_logits)
                TPN_target = torch.transpose(target_support, 0, 1)
                TPN_target = TPN_target.reshape(
                    -1,
                )
                acc1, acc3 = self.accuracy(TPN_pred, TPN_target, topk=(1, 1))
                log = f"\tmeta-train [{i}/{self.cfg.task_steps}] Loss: {s_loss.item():.4f}, Acc(1): {acc1.item():.2f}"
                logs.append(log)
                print(log)

        return fast_weights

    def finetune(
        self, rank, net, train_loader, criterion, optimizer, epoch, num_epochs, logs
    ):
        net.eval()

        for batch_idx, data in enumerate(train_loader):
            features = data[0].cuda()
            targets = data[3].cuda()
            targets = targets.type(torch.LongTensor).cuda() - 1

            logits = net(features)
            loss = criterion(logits, targets)

            if rank == 0:
                if batch_idx % self.cfg.log_freq == 0:
                    acc1, acc5 = self.accuracy(logits, targets, topk=(1, 3))
                    log = f"Epoch [{epoch + 1}/{num_epochs}]-({batch_idx}/{len(train_loader)}) "
                    log += f"\tLoss: {loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}"
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
                targets = targets.type(torch.LongTensor).cuda()

                logits = net(features)

                total_loss += criterion(logits, targets)
                total_targets.append(targets)
                total_logits.append(logits)

            if len(total_targets) > 0:
                total_targets = torch.cat(total_targets, dim=0)
                total_logits = torch.cat(total_logits, dim=0)
                acc1, acc5 = self.accuracy(total_logits, total_targets, topk=(1, 3))
                f1, recall, precision = self.scores(total_logits, total_targets)
            total_loss /= len(val_loader)

            if rank == 0:
                log = f"[Finetune] Validation Loss: {total_loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}"
                log += f", F1: {f1.item():.2f}, Recall: {recall.item():.2f}, Precision: {precision.item():.2f}"
                logs.append(log)
                print(log)

    def visualize(self, rank, net, val_loader, criterion, logs, writer, tag):
        net.eval()

        total_targets = []
        total_logits = []
        for batch_idx, data in enumerate(val_loader):
            features = data[0].cuda()
            targets = data[3].cuda()

            logits = net.base_model(features)

            # print(logits.shape)
            # print(targets.shape)

            total_targets.append(targets)
            total_logits.append(logits.view(targets.shape[0], -1))

        total_targets = torch.cat(total_targets, dim=0)
        total_logits = torch.cat(total_logits, dim=0)
        writer.add_embedding(
            total_logits, metadata=total_targets, global_step=0, tag="feature"
        )

    def save_checkpoint(self, filename, epoch, state_dict, optimizer):
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        state = {
            "epoch": epoch + 1,
            "state_dict": state_dict,
            "optimizer": optimizer.state_dict(),
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
            f1 = metrics.f1_score(target_val, out_val, average="macro", zero_division=0)
            acc = metrics.accuracy_score(target_val, out_val)

            return f1, recall, precision
