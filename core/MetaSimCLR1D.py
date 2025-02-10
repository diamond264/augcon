import os
import copy
import random
import wandb

from collections import defaultdict
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

# from torch.utils.tensorboard import SummaryWriter


class Adversary_Negatives(nn.Module):
    def __init__(self, bank_size, dim):
        super(Adversary_Negatives, self).__init__()
        self.vars = nn.ParameterList()
        W = torch.randn(bank_size, dim)
        self.vars.append(W)

    def init(self, net, dataset):
        with torch.no_grad():
            net.eval()
            shuffled_dataset = torch.utils.data.Subset(
                dataset, torch.randperm(len(dataset))
            )
            for i, (x, _, _, _, _) in enumerate(shuffled_dataset):
                if i >= len(self.vars[0]):
                    break
                x = x.cuda()
                z = net.get_z(x)
                self.vars[0][i] = z

    def forward(self, vars=None):
        if vars is None:
            vars = self.vars

        return vars[0]

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


class Encoder(nn.Module):
    def __init__(self, input_channels, z_dim, num_layers):
        super(Encoder, self).__init__()
        self.vars = nn.ParameterList()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.num_blocks = num_layers
        if self.num_blocks == 3:
            in_dims = [input_channels, 32, 64]
            out_dims = [32, 64, z_dim]
            kernel_sizes = [24, 16, 8]

        elif self.num_blocks == 4:
            in_dims = [input_channels, 32, 64, 128]
            out_dims = [32, 64, 128, z_dim]
            kernel_sizes = [24, 16, 12, 8]

        elif self.num_blocks == 5:
            in_dims = [input_channels, 32, 64, 128, 256]
            out_dims = [32, 64, 128, 256, z_dim]
            kernel_sizes = [24, 16, 12, 8, 6]

        elif self.num_blocks == 6:
            in_dims = [input_channels, 32, 64, 128, 256, 512]
            out_dims = [32, 64, 128, 256, 512, z_dim]
            kernel_sizes = [24, 16, 12, 8, 6, 4]

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
        for _ in range(self.num_blocks):
            w, b = vars[idx], vars[idx + 1]
            idx += 2
            x = F.conv1d(x, w, b)
            x = self.relu(x)
            x = self.dropout(x)

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
    def __init__(self, in_dim=96, hidden_1=256, hidden_2=128, out_dim=50):
        super(Classifier, self).__init__()
        self.vars = nn.ParameterList()

        self.relu = nn.ReLU()

        fc1 = nn.Linear(in_dim, hidden_1)
        w = fc1.weight
        b = fc1.bias
        self.vars.append(w)
        self.vars.append(b)

        fc2 = nn.Linear(hidden_1, hidden_2)
        w = fc2.weight
        b = fc2.bias
        self.vars.append(w)
        self.vars.append(b)

        fc3 = nn.Linear(hidden_2, out_dim)
        w = fc3.weight
        b = fc3.bias
        self.vars.append(w)
        self.vars.append(b)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars

        x = F.linear(x, vars[0], vars[1])
        x = self.relu(x)
        x = F.linear(x, vars[2], vars[3])
        x = self.relu(x)
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

    def __init__(self, input_channels=3, z_dim=96, out_dim=50, T=0.1, num_layers=3, mlp=True):
        super(SimCLRNet, self).__init__()
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder = Encoder(input_channels, z_dim, num_layers)

        self.mlp = mlp
        if mlp:  # hack: brute-force replacement
            self.classifier = Classifier(in_dim=z_dim, out_dim=out_dim)

    def get_z(self, x):
        z = self.encoder(x)
        if self.mlp:
            z = self.classifier(z)
        return z

    def forward(self, feature, aug_feature, vars=None):
        if vars is None:
            vars = nn.ParameterList()
            vars.extend(self.encoder.parameters())
            if self.mlp:
                vars.extend(self.classifier.parameters())

        enc_vars = vars[: len(self.encoder.parameters())]
        if self.mlp:
            cls_vars = vars[len(self.encoder.parameters()) :]

        z = self.encoder(feature, enc_vars)
        if self.mlp:
            z = self.classifier(z, cls_vars)
        z = F.normalize(z, dim=1)

        aug_z = self.encoder(aug_feature, enc_vars)
        if self.mlp:
            aug_z = self.classifier(aug_z, cls_vars)
        aug_z = F.normalize(aug_z, dim=1)

        LARGE_NUM = 1e9
        masks = F.one_hot(torch.arange(z.size(0)), z.size(0)).cuda()

        pos_logits = torch.matmul(z, aug_z.t())
        neg_logits_1 = torch.matmul(z, z.t())
        neg_logits_1 = neg_logits_1 - masks * LARGE_NUM
        neg_logits_2 = torch.matmul(aug_z, aug_z.t())
        neg_logits_2 = neg_logits_2 - masks * LARGE_NUM

        logits = torch.cat([pos_logits, neg_logits_1, neg_logits_2], dim=1)
        # logits = torch.cat([pos_logits], dim=1)
        logits /= self.T
        labels = torch.arange(z.size(0)).cuda()

        return logits, labels

    def forward_detached(self, feature, aug_feature, vars=None):
        if vars is None:
            vars = nn.ParameterList()
            vars.extend(self.encoder.parameters())
            if self.mlp:
                vars.extend(self.classifier.parameters())

        enc_vars = vars[: len(self.encoder.parameters())]
        if self.mlp:
            cls_vars = vars[len(self.encoder.parameters()) :]

        z = self.encoder(feature, enc_vars)
        if self.mlp:
            z = self.classifier(z, cls_vars)
        z = F.normalize(z, dim=1).detach()

        aug_z = self.encoder(aug_feature, enc_vars)
        if self.mlp:
            aug_z = self.classifier(aug_z, cls_vars)
        aug_z = F.normalize(aug_z, dim=1).detach()

        LARGE_NUM = 1e9
        masks = F.one_hot(torch.arange(z.size(0)), z.size(0)).cuda()

        pos_logits = torch.matmul(z, aug_z.t())
        # neg_logits_1 = torch.matmul(z, z.t())
        # neg_logits_1 = neg_logits_1 - masks * LARGE_NUM
        # neg_logits_2 = torch.matmul(aug_z, aug_z.t())
        # neg_logits_2 = neg_logits_2 - masks * LARGE_NUM

        # logits = torch.cat([pos_logits, neg_logits_1, neg_logits_2], dim=1)
        logits = torch.cat([pos_logits], dim=1)
        logits /= self.T
        labels = torch.arange(z.size(0)).cuda()

        return logits, labels

    def forward_wo_nq(self, feature, aug_feature, neg_feature, vars=None):
        if vars is None:
            vars = nn.ParameterList()
            vars.extend(self.encoder.parameters())
            if self.mlp:
                vars.extend(self.classifier.parameters())

        enc_vars = vars[: len(self.encoder.parameters())]
        if self.mlp:
            cls_vars = vars[len(self.encoder.parameters()) :]

        z = self.encoder(feature, enc_vars)
        if self.mlp:
            z = self.classifier(z, cls_vars)
        z = F.normalize(z, dim=1)

        aug_z = self.encoder(aug_feature, enc_vars)
        if self.mlp:
            aug_z = self.classifier(aug_z, cls_vars)
        aug_z = F.normalize(aug_z, dim=1)

        LARGE_NUM = 1e9
        masks = F.one_hot(torch.arange(z.size(0)), z.size(0)).cuda()

        pos_logits = torch.matmul(z, aug_z.t())
        neg_logits_1 = torch.matmul(z, z.t())
        neg_logits_1 = neg_logits_1 - masks * LARGE_NUM
        neg_logits_2 = torch.matmul(aug_z, aug_z.t())
        neg_logits_2 = neg_logits_2 - masks * LARGE_NUM

        logits = torch.cat([pos_logits, neg_logits_1, neg_logits_2], dim=1)
        logits /= self.T
        labels = torch.arange(z.size(0)).cuda()

        return logits, labels

    def forward_w_nq(self, feature, aug_feature, neg_feature, vars=None):
        if vars is None:
            vars = nn.ParameterList()
            vars.extend(self.encoder.parameters())
            if self.mlp:
                vars.extend(self.classifier.parameters())

        enc_vars = vars[: len(self.encoder.parameters())]
        if self.mlp:
            cls_vars = vars[len(self.encoder.parameters()) :]

        z = self.encoder(feature, enc_vars)
        if self.mlp:
            z = self.classifier(z, cls_vars)
        z = F.normalize(z, dim=1)

        aug_z = self.encoder(aug_feature, enc_vars)
        if self.mlp:
            aug_z = self.classifier(aug_z, cls_vars)
        aug_z = F.normalize(aug_z, dim=1)

        # neg_z = self.encoder(neg_feature, enc_vars)
        # if self.mlp:
        #     neg_z = self.classifier(neg_z, cls_vars)
        neg_z = neg_feature
        neg_z = F.normalize(neg_z, dim=1)

        LARGE_NUM = 1e9
        masks = F.one_hot(torch.arange(z.size(0)), z.size(0)).cuda()

        pos_logits = torch.matmul(z, aug_z.t())
        # pos_logits = torch.einsum("ij,ij->i", z, aug_z)
        # pos_logits = torch.unsqueeze(pos_logits, dim=1)
        neg_logits_1 = torch.matmul(z, z.t())
        neg_logits_1 = neg_logits_1 - masks * LARGE_NUM
        neg_logits_2 = torch.matmul(aug_z, aug_z.t())
        neg_logits_2 = neg_logits_2 - masks * LARGE_NUM

        neg_logits_3 = torch.matmul(z, neg_z.t())
        # neg_logits_4 = torch.matmul(aug_z, neg_z.t())
        # print(pos_logits[0][1])
        # print(neg_logits_2[0][1])
        # wandb.log(
        #     {
        #         "pos logits 1 max": torch.mean(torch.max(pos_logits, dim=1)[0]).item(),
        #         "pos logits 1 mean": torch.mean(
        #             torch.mean(pos_logits, dim=1)[0]
        #         ).item(),
        #         "neg logits 3 max": torch.mean(
        #             torch.max(neg_logits_3, dim=1)[0]
        #         ).item(),
        #         "neg logits 3 mean": torch.mean(
        #             torch.mean(neg_logits_3, dim=1)[0]
        #         ).item(),
        #     }
        # )
        # print(torch.mean(torch.max(neg_logits_1, dim=1)[0]))
        # print(torch.mean(torch.mean(neg_logits_1, dim=1)[1]))
        # print(torch.mean(torch.max(neg_logits_3, dim=1)[0]))
        # print(torch.mean(torch.mean(neg_logits_3, dim=1)[1]))

        logits = torch.cat(
            [pos_logits, neg_logits_1, neg_logits_2, neg_logits_3],
            # [pos_logits, neg_logits_3],
            dim=1,
        )
        logits /= self.T
        labels = torch.arange(z.size(0)).cuda()
        # labels = torch.zeros(z.size(0), dtype=torch.long).cuda()

        return logits, labels

    def forward_w_nq_detached(self, feature, aug_feature, neg_feature, vars=None):
        if vars is None:
            vars = nn.ParameterList()
            vars.extend(self.encoder.parameters())
            if self.mlp:
                vars.extend(self.classifier.parameters())

        enc_vars = vars[: len(self.encoder.parameters())]
        if self.mlp:
            cls_vars = vars[len(self.encoder.parameters()) :]

        z = self.encoder(feature, enc_vars)
        if self.mlp:
            z = self.classifier(z, cls_vars)
        z = F.normalize(z, dim=1).detach()

        aug_z = self.encoder(aug_feature, enc_vars)
        if self.mlp:
            aug_z = self.classifier(aug_z, cls_vars)
        aug_z = F.normalize(aug_z, dim=1).detach()

        # neg_z = self.encoder(neg_feature, enc_vars)
        # if self.mlp:
        #     neg_z = self.classifier(neg_z, cls_vars)
        neg_z = neg_feature
        neg_z = F.normalize(neg_z, dim=1)

        LARGE_NUM = 1e9
        masks = F.one_hot(torch.arange(z.size(0)), z.size(0)).cuda()

        pos_logits = torch.matmul(z, aug_z.t())
        # pos_logits = torch.einsum("ij,ij->i", z, aug_z)
        # pos_logits = torch.unsqueeze(pos_logits, dim=1)
        neg_logits_1 = torch.matmul(z, z.t())
        neg_logits_1 = neg_logits_1 - masks * LARGE_NUM
        neg_logits_2 = torch.matmul(aug_z, aug_z.t())
        neg_logits_2 = neg_logits_2 - masks * LARGE_NUM

        neg_logits_3 = torch.matmul(z, neg_z.t())
        # neg_logits_4 = torch.matmul(aug_z, neg_z.t())

        logits = torch.cat(
            [pos_logits, neg_logits_1, neg_logits_2, neg_logits_3],
            # [pos_logits, neg_logits_3],
            dim=1,
        )
        logits /= self.T
        labels = torch.arange(z.size(0)).cuda()
        # labels = torch.zeros(z.size(0), dtype=torch.long).cuda()

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
    def __init__(self, input_channels, z_dim, num_cls, num_layers, mlp=True):
        super(SimCLRClassifier, self).__init__()
        self.base_model = Encoder(input_channels, z_dim, num_layers)
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
        # wandb.init(
        #     project="MetaSimCLR1D",
        #     config={
        #         "pretext": self.cfg.pretext,
        #         "mode": self.cfg.mode,
        #         "train_dataset_path": self.cfg.train_dataset_path,
        #         "test_dataset_path": self.cfg.test_dataset_path,
        #         "epochs": self.cfg.epochs,
        #         "batch_size": self.cfg.batch_size,
        #         "lr": self.cfg.lr,
        #         "wd": self.cfg.wd,
        #         "pretrained": self.cfg.pretrained,
        #         "task_per_domain": self.cfg.task_per_domain,
        #         "num_task": self.cfg.num_task,
        #         "multi_cond_num_task": self.cfg.multi_cond_num_task,
        #         "task_size": self.cfg.task_size,
        #         "task_lr": self.cfg.task_lr,
        #         "task_steps": self.cfg.task_steps,
        #         "domain_adaptation": self.cfg.domain_adaptation,
        #         "adapt_w_neg": self.cfg.adapt_w_neg,
        #         "bank_size": self.cfg.bank_size,
        #         "membank_lr": self.cfg.membank_lr,
        #         "membank_m": self.cfg.membank_m,
        #         "membank_wd": self.cfg.membank_wd,
        #         "task_membank_lr": self.cfg.task_membank_lr,
        #     },
        # )
        # if self.cfg.domain_adaptation:
        #     setting = "domain_adaptation"
        #     if self.cfg.adapt_w_neg:
        #         setting = "domain_adaptation_w_neg"
        # else:
        #     setting = "no_domain_adaptation"
        # wandb.run.name = setting

        num_gpus = len(self.gpu)
        logs = mp.Manager().list([])
        self.logger.info("Executing SimCLR")
        self.logger.info("Logs are skipped during training")
        if num_gpus > 1:
            mp.spawn(
                self.main_worker,
                args=(
                    num_gpus,
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    logs,
                ),
                nprocs=num_gpus,
            )
        else:
            self.main_worker(0, 1, train_dataset, val_dataset, test_dataset, logs)

        for log in logs:
            self.logger.info(log)

    def log(self, rank, logs, log_txt):
        if rank == 0:
            logs.append(log_txt)
            print(log_txt)

    def main_worker(
        self,
        rank,
        world_size,
        train_dataset,
        val_dataset,
        test_dataset,
        logs,
    ):
        # Model initialization
        net = SimCLRNet(
            self.cfg.input_channels, self.cfg.z_dim, self.cfg.out_dim, self.cfg.T, self.cfg.num_layers, True
        )
        if self.cfg.mode == "finetune" or self.cfg.mode == "eval_finetune":
            cls_net = SimCLRClassifier(
                self.cfg.input_channels, self.cfg.z_dim, self.cfg.num_cls, self.cfg.num_layers, self.cfg.mlp
            )

        num_params = sum(p.numel() for p in net.parameters())
        print(f"Number of parameters : {num_params}")
        # exit()

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

            train_sampler = DistributedSampler(train_dataset)
            meta_train_dataset = torch.utils.data.Subset(
                train_dataset, list(train_sampler)
            )

            if self.cfg.mode == "finetune" or self.cfg.mode == "eval_finetune":
                cls_net.cuda()
                cls_net = nn.parallel.DistributedDataParallel(
                    cls_net, device_ids=[rank], find_unused_parameters=True
                )

                # if not neg_dataset is None:
                #     neg_sampler = DistributedSampler(neg_dataset)
                #     meta_neg_dataset = torch.utils.data.Subset(
                #         neg_dataset, list(neg_sampler)
                #     )

                if len(val_dataset) > 0:
                    val_sampler = DistributedSampler(val_dataset)
                test_sampler = DistributedSampler(test_dataset)

                # collate_fn = subject_collate if self.cfg.mode == 'pretrain' else None
                collate_fn = None
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.cfg.batch_size // world_size,
                    shuffle=False,
                    sampler=train_sampler,
                    collate_fn=collate_fn,
                    num_workers=self.cfg.num_workers,
                    drop_last=True,
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.cfg.batch_size // world_size,
                    shuffle=False,
                    sampler=test_sampler,
                    collate_fn=collate_fn,
                    num_workers=self.cfg.num_workers,
                    drop_last=True,
                )
                if len(val_dataset) > 0:
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=self.cfg.batch_size // world_size,
                        shuffle=False,
                        sampler=val_sampler,
                        collate_fn=collate_fn,
                        num_workers=self.cfg.num_workers,
                        drop_last=True,
                    )
            self.log(rank, logs, f"Using DDP ({len(list(train_sampler))} per worker)")
        else:
            torch.cuda.set_device(rank)
            net.cuda()

            meta_train_dataset = train_dataset

            if self.cfg.mode == "finetune" or self.cfg.mode == "eval_finetune":
                cls_net.cuda()

                # if not neg_dataset is None:
                #     meta_neg_dataset = neg_dataset

                # collate_fn = subject_collate if self.cfg.mode == 'pretrain' else None
                collate_fn = None
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=self.cfg.num_workers,
                    drop_last=True,
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=self.cfg.num_workers,
                    drop_last=True,
                )
                if len(val_dataset) > 0:
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=self.cfg.batch_size,
                        shuffle=True,
                        collate_fn=collate_fn,
                        num_workers=self.cfg.num_workers,
                        drop_last=True,
                    )
            self.log(rank, logs, f"Using single GPU ({len(train_dataset)} per worker)")

        indices_per_domain = self.split_per_domain(meta_train_dataset)

        memory_bank = Adversary_Negatives(self.cfg.bank_size, self.cfg.out_dim).cuda()
        memory_bank.init(net, meta_train_dataset)

        # Define criterion
        if self.cfg.criterion == "crossentropy":
            criterion = nn.CrossEntropyLoss().cuda()

        # For finetuning, load pretrained model
        if self.cfg.mode == "finetune":
            if os.path.isfile(self.cfg.pretrained):
                self.log(
                    rank, logs, f"Loading pretrained model ({self.cfg.pretrained})"
                )
                loc = "cuda:{}".format(rank)
                state = torch.load(self.cfg.pretrained, map_location=loc)
                if self.cfg.pretext != "setsimclr":
                    state = torch.load(self.cfg.pretrained, map_location=loc)[
                        "state_dict"
                    ]

                new_state = {}
                if self.cfg.no_vars:  # option for debugging
                    for i, (k, v) in enumerate(state.items()):
                        new_k = list(net.state_dict().keys())[i]
                        new_state[new_k] = v
                else:
                    new_state = state

                msg = net.load_state_dict(new_state, strict=False)
                self.log(rank, logs, f"Missing keys: {msg.missing_keys}")
            else:
                self.log(rank, logs, f"No checkpoint found at '{self.cfg.pretrained}'")

            if os.path.isfile(self.cfg.pretrained_membank):
                self.log(
                    rank,
                    logs,
                    f"Loading pretrained membank ({self.cfg.pretrained_membank})",
                )
                loc = "cuda:{}".format(rank)
                membank_state = torch.load(
                    self.cfg.pretrained_membank, map_location=loc
                )
                msg = memory_bank.load_state_dict(membank_state["state_dict"])
                self.log(rank, logs, f"Missing keys: {msg.missing_keys}")

            # Meta-train the pretrained model for domain adaptation
            if world_size > 1:
                train_sampler.set_epoch(0)
                # for debugging
                # if not neg_dataset is None:
                #     neg_sampler.set_epoch(0)

            shuffled_idx = torch.randperm(len(meta_train_dataset))
            meta_train_dataset = torch.utils.data.Subset(
                meta_train_dataset, shuffled_idx
            )

            # # for debugging
            # st = 200
            # shuffled_neg_idx = torch.randperm(len(meta_neg_dataset))[
            #     st : st + self.cfg.num_negs
            # ]
            # meta_neg_dataset = torch.utils.data.Subset(
            #     meta_neg_dataset, shuffled_neg_idx
            # )

            net.eval()
            enc_parameters = list(copy.deepcopy(net.parameters()))

            # perform domgin adaptation
            if self.cfg.domain_adaptation:
                self.log(rank, logs, f"Perform domain adaptation step")
                support = [e[1] for e in meta_train_dataset]
                support = torch.stack(support, dim=0).cuda()

                pos_support = [e[2] for e in meta_train_dataset]
                pos_support = torch.stack(pos_support, dim=0).cuda()

                # # for debugging
                # neg_support = [e[1] for e in meta_neg_dataset]
                # neg_support = torch.stack(neg_support, dim=0).cuda()

                # for debugging
                test_support = [e[1] for e in val_dataset]
                test_support = torch.stack(test_support, dim=0).cuda()
                test_pos_support = [e[2] for e in val_dataset]
                test_pos_support = torch.stack(test_pos_support, dim=0).cuda()

                enc_parameters, _ = self.meta_train(
                    rank,
                    net,
                    support,
                    pos_support,
                    # test_support,
                    # test_pos_support,
                    memory_bank,
                    criterion,
                    log_steps=True,
                    logs=logs,
                )

            self.log(rank, logs, "Loading encoder parameters to the classifier")
            enc_dict = {}
            for idx, k in enumerate(list(cls_net.state_dict().keys())):
                if not "classifier" in k:
                    # k_ = k.replace("encoder.", "base_model.")
                    enc_dict[k] = enc_parameters[idx]
                else:
                    enc_dict[k] = cls_net.state_dict()[k]

            msg = cls_net.load_state_dict(enc_dict, strict=True)
            self.log(rank, logs, f"Missing keys: {msg.missing_keys}")

            # if self.cfg.visualization:
            #     is_meta = True if "meta" in self.cfg.pretrained else False
            #     tag = f"{self.cfg.dataset_name}_{self.cfg.pretext}_{self.cfg.domain_adaptation}_meta{is_meta}"
            #     writer = SummaryWriter(f"./logs/{tag}")
            #     self.visualize(rank, cls_net, test_loader, criterion, logs, writer, tag)

            # now replace the encoder with the classifier
            net = cls_net
            # Freezing the encoder
            if self.cfg.freeze:
                self.log(rank, logs, "Freezing the encoder")
                for name, param in net.named_parameters():
                    if not "classifier" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

        if self.cfg.membank_optimizer == "sgd":
            membank_optimizer = torch.optim.SGD(
                memory_bank.parameters(),
                self.cfg.membank_lr,
                momentum=self.cfg.membank_m,
                weight_decay=self.cfg.membank_wd,
            )
        if self.cfg.membank_optimizer == "adam":
            membank_optimizer = torch.optim.Adam(
                memory_bank.parameters(),
                self.cfg.membank_lr,
                weight_decay=self.cfg.membank_wd,
            )

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
        # if self.cfg.mode == 'finetune':
        #     scheduler = StepLR(optimizer, step_size=self.cfg.lr_decay_step, gamma=self.cfg.lr_decay)

        # Load checkpoint if exists
        if os.path.isfile(self.cfg.resume):
            self.log(rank, logs, f"Resume from checkpoint - {self.cfg.resume}")
            loc = "cuda:{}".format(rank)
            state = torch.load(self.cfg.resume, map_location=loc)

            for k, v in list(state["state_dict"].items()):
                new_k = k
                if world_size > 1:
                    new_k = "module." + k
                if new_k in net.state_dict().keys():
                    state["state_dict"][new_k] = v
                if k not in net.state_dict().keys():
                    state["state_dict"].pop(k, None)

            net.load_state_dict(state["state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            self.cfg.start_epoch = state["epoch"]

        # Handling the modes (train or eval)
        # if self.cfg.mode == "eval_pretrain":
        #     self.validate_pretrain(rank, net, test_loader, criterion, logs)
        if self.cfg.mode == "eval_finetune":
            self.validate_finetune(rank, net, test_loader, criterion, logs)
        else:  # pretrain or finetune
            # loss_best = 0
            for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
                if world_size > 1:
                    train_sampler.set_epoch(epoch)
                    # if not neg_dataset is None:
                    #     neg_sampler.set_epoch(epoch)
                    if len(val_dataset) > 0:
                        val_sampler.set_epoch(epoch)
                    test_sampler.set_epoch(epoch)

                if self.cfg.mode == "pretrain":
                    supports = []
                    queries = []
                    pos_supports = []
                    pos_queries = []
                    if self.cfg.task_per_domain:
                        supports, pos_supports, queries, pos_queries = (
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
                            multi_cond_pos_supports,
                            multi_cond_queries,
                            multi_cond_pos_queries,
                        ) = self.gen_random_tasks(
                            meta_train_dataset,
                            self.cfg.task_size,
                            self.cfg.multi_cond_num_task,
                        )
                        supports = supports + multi_cond_supports
                        queries = queries + multi_cond_queries
                        pos_supports = pos_supports + multi_cond_pos_supports
                        pos_queries = pos_queries + multi_cond_pos_queries

                    self.pretrain(
                        rank,
                        net,
                        supports,
                        pos_supports,
                        queries,
                        pos_queries,
                        memory_bank,
                        criterion,
                        optimizer,
                        membank_optimizer,
                        epoch,
                        self.cfg.epochs,
                        logs,
                    )

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

                elif self.cfg.mode == "finetune":
                    self.finetune(
                        rank,
                        net,
                        train_loader,
                        criterion,
                        optimizer,
                        epoch,
                        self.cfg.epochs,
                        logs,
                    )

                    # if len(val_dataset) > 0:
                    #     self.validate_finetune(rank, net, val_loader, criterion, logs)

                if rank == 0 and (epoch + 1) % self.cfg.save_freq == 0:
                    ckpt_dir = self.cfg.ckpt_dir
                    ckpt_filename = "checkpoint_{:04d}.pth.tar".format(epoch)
                    ckpt_filename = os.path.join(ckpt_dir, ckpt_filename)
                    state_dict = net.state_dict()
                    membank_filename = "membank_{:04d}.pth.tar".format(epoch)
                    membank_filename = os.path.join(ckpt_dir, membank_filename)
                    membank_state_dict = memory_bank.state_dict()
                    if world_size > 1:
                        for k, v in list(state_dict.items()):
                            if "module." in k:
                                state_dict[k.replace("module.", "")] = v
                                del state_dict[k]
                    self.save_checkpoint(ckpt_filename, epoch, state_dict, optimizer)
                    self.save_checkpoint(
                        membank_filename, epoch, membank_state_dict, membank_optimizer
                    )

            if self.cfg.mode == "finetune":
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

    def gen_per_domain_tasks(
        self, dataset, indices_per_domain, task_size, num_task=None
    ):
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

                    query_ = torch.utils.data.Subset(
                        dataset, indices[task_size : 2 * task_size]
                    )
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
                        support = (
                            support * (task_size // len(support))
                            + support[: task_size % len(support)]
                        )
                        pos_support = (
                            pos_support * (task_size // len(pos_support))
                            + pos_support[: task_size % len(pos_support)]
                        )
                    support = torch.stack(support, dim=0)
                    pos_support = torch.stack(pos_support, dim=0)

                    query_ = torch.utils.data.Subset(
                        dataset, indices[task_size : 2 * task_size]
                    )
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
                        query = (
                            query * (task_size // len(query))
                            + query[: task_size % len(query)]
                        )
                        pos_query = (
                            pos_query * (task_size // len(pos_query))
                            + pos_query[: task_size % len(pos_query)]
                        )
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

    def pretrain(
        self,
        rank,
        net,
        supports,
        pos_supports,
        queries,
        pos_queries,
        memory_bank,
        criterion,
        optimizer,
        membank_optimizer,
        epoch,
        num_epochs,
        logs,
    ):
        net.train()
        net.zero_grad()

        if self.cfg.adapt_w_neg:
            q_losses_enc = []
            q_losses_mem = []
        else:
            q_losses = []
        num_tasks = len(supports)
        for task_idx in range(num_tasks):
            support = supports[task_idx].cuda()
            pos_support = pos_supports[task_idx].cuda()
            query = queries[task_idx].cuda()
            pos_query = pos_queries[task_idx].cuda()

            fast_weights, fast_negatives = self.meta_train(
                rank,
                net,
                support,
                pos_support,
                memory_bank,
                criterion,
                log_steps=self.cfg.log_meta_train,
                logs=logs,
            )

            if self.cfg.adapt_w_neg:
                # negatives_enc = memory_bank(fast_negatives).detach()
                # q_logits_enc, q_targets_enc = net.forward_w_nq(
                #     support, pos_support, negatives_enc, fast_weights
                # )
                q_logits_enc, q_targets_enc = net.forward(
                    query, pos_query, fast_weights
                )
                # negatives_enc = memory_bank(fast_negatives)
                negatives_enc = memory_bank()
                q_logits_mem, q_targets_mem = net.forward_w_nq_detached(
                    query, pos_query, negatives_enc, fast_weights
                )
                q_loss_enc = criterion(q_logits_enc, q_targets_enc)
                q_loss_mem = -criterion(q_logits_mem, q_targets_mem)
                q_losses_enc.append(q_loss_enc)
                q_losses_mem.append(q_loss_mem)

                if task_idx % self.cfg.log_freq == 0:
                    acc = self.accuracy(q_logits_enc, q_targets_enc, topk=(1, 5))
                    log = f"Epoch [{epoch+1}/{num_epochs}]  \t({task_idx}/{len(supports)}) "
                    log += f"Loss: {q_loss_enc.item():.4f}, Acc(1): {acc[0].item():.2f}, Acc(5): {acc[1].item():.2f}"
                    self.log(rank, logs, log)

            else:
                q_logits, q_targets = net(query, pos_query, fast_weights)
                q_loss = criterion(q_logits, q_targets)
                q_losses.append(q_loss)

                if task_idx % self.cfg.log_freq == 0:
                    acc = self.accuracy(q_logits, q_targets, topk=(1, 5))
                    log = f"Epoch [{epoch+1}/{num_epochs}]  \t({task_idx}/{len(supports)}) "
                    log += f"Loss: {q_loss.item():.4f}, Acc(1): {acc[0].item():.2f}, Acc(5): {acc[1].item():.2f}"
                    self.log(rank, logs, log)

        if self.cfg.adapt_w_neg:
            q_losses_enc = torch.stack(q_losses_enc, dim=0)
            q_losses_mem = torch.stack(q_losses_mem, dim=0)
            loss_enc = torch.sum(q_losses_enc)
            loss_mem = torch.sum(q_losses_mem)
            loss_enc = loss_enc / len(supports)
            loss_mem = loss_mem / len(supports)

            optimizer.zero_grad()
            loss_enc.backward()
            optimizer.step()

            membank_optimizer.zero_grad()
            loss_mem.backward()
            membank_optimizer.step()

            # with torch.no_grad():
            #     query = queries[0].cuda()
            #     pos_query = pos_queries[0].cuda()
            #     negatives_enc = memory_bank()
            #     q_logits_mem, q_targets_mem = net.forward_w_nq_detached(
            #         query, pos_query, negatives_enc, fast_weights
            #     )
            #     q_loss_mem = -criterion(q_logits_mem, q_targets_mem)

            #     acc = self.accuracy(q_logits_mem, q_targets_mem, topk=(1, 5))
            #     log = f"Epoch [{epoch+1}/{num_epochs}]  \t({task_idx}/{len(supports)}) "
            #     log += f"Loss: {q_loss_mem.item():.4f}, Acc(1): {acc[0].item():.2f}, Acc(5): {acc[1].item():.2f}"
            #     self.log(rank, logs, log)

        else:
            q_losses = torch.stack(q_losses, dim=0)
            loss = torch.sum(q_losses)
            loss = loss / len(supports)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # self.log(rank, logs, f"Epoch [{epoch+1}/{num_epochs}] {loss.item():.4f}")

    def meta_train(
        self,
        rank,
        net,
        support,
        pos_support,
        # test_support,
        # test_pos_support,
        memory_bank,
        criterion,
        log_steps=False,
        logs=None,
    ):
        # fast_weights = list(copy.deepcopy(net.parameters()))
        fast_weights = list(net.parameters())
        fast_negatives = list(memory_bank.parameters())
        # support = support[:5]
        # pos_support = pos_support[:5]

        for i in range(self.cfg.task_steps):
            # s_logits, s_targets = net(support, pos_support, fast_weights)
            batch_size = len(support)
            # batch_size = 16
            for j in range(len(support) // batch_size):
                support_batch = support[j * batch_size : (j + 1) * batch_size]
                pos_support_batch = pos_support[j * batch_size : (j + 1) * batch_size]
                # print(support_batch.shape)
                if self.cfg.adapt_w_neg:
                    negatives = memory_bank(fast_negatives).detach()
                    s_logits, s_targets = net.forward_w_nq(
                        support_batch, pos_support_batch, negatives, fast_weights
                    )

                    s_loss = criterion(s_logits, s_targets)
                    grad = torch.autograd.grad(s_loss, fast_weights)
                    fast_weights = list(
                        map(
                            lambda p: p[1] - self.cfg.task_lr * p[0],
                            zip(grad, fast_weights),
                        )
                    )

                    # negatives = memory_bank(fast_negatives)
                    # s_logits, s_targets = net.forward_w_nq_detached(
                    #     support_batch, pos_support_batch, negatives, fast_weights
                    # )

                    # membank_loss = -criterion(s_logits, s_targets)
                    # membank_grad = torch.autograd.grad(membank_loss, fast_negatives)
                    # fast_negatives = list(
                    #     map(
                    #         lambda p: p[1] - self.cfg.task_membank_lr * p[0],
                    #         zip(membank_grad, fast_negatives),
                    #     )
                    # )

                else:
                    # n = 90
                    # m = 150
                    support_ = support_batch
                    pos_support_ = pos_support_batch
                    # support_ = test_support
                    # pos_support_ = test_pos_support
                    # negatives = support[n:m]
                    # negatives = test_support[:m]
                    # pos_negatives = test_pos_support[:m]
                    # support_ = torch.cat([support_, negatives], dim=0)
                    # pos_support_ = torch.cat([pos_support_, pos_negatives], dim=0)
                    # s_logits, s_targets = net.forward_w_nq(
                    #     support_, pos_support_, negatives, fast_weights
                    # )
                    s_logits, s_targets = net.forward(
                        support_, pos_support_, fast_weights
                    )
                    # print(s_logits.shape)
                    # assert 0
                    s_loss = criterion(s_logits, s_targets)

                    grad = torch.autograd.grad(s_loss, fast_weights)
                    fast_weights = list(
                        map(
                            lambda p: p[1] - self.cfg.task_lr * p[0],
                            zip(grad, fast_weights),
                        )
                    )

            # with torch.no_grad():
            #     k = 50
            #     s_logits, s_targets = net(
            #         support[:k],
            #         pos_support[:k],
            #         fast_weights,
            #     )
            #     s_loss = criterion(s_logits, s_targets)

            # k = 50
            # k_logits, k_targets = net(
            #     test_support[:k],
            #     test_pos_support[:k],
            #     fast_weights,
            # )
            # k_loss = criterion(k_logits, k_targets)

            # if log_steps:
            #     acc = self.accuracy(k_logits, k_targets, topk=(1, 5))
            #     log = f"\tmeta-train [{i}/{self.cfg.task_steps}] Loss_train: {s_loss.item():.4f}, "
            #     log += f"Loss_test: {k_loss.item():.4f}, "
            #     log += f"Acc(1): {acc[0].item():.2f}, Acc(5): {acc[1].item():.2f}"
            #     self.log(rank, logs, log)
            # # s_logits, s_targets = net(support, pos_support, fast_weights)
            # s_logits, s_targets = net(test_support, test_pos_support, fast_weights)
            # s_loss = criterion(s_logits, s_targets)

            if log_steps:
                acc = self.accuracy(s_logits, s_targets, topk=(1, 1))
                log = f"\tmeta-train [{i}/{self.cfg.task_steps}] Loss: {s_loss.item():.4f}, "
                log += f"Acc(1): {acc[0].item():.2f}, Acc(5): {acc[1].item():.2f}"
                self.log(rank, logs, log)
                # wandb.log(
                #     {"meta-train loss": s_loss.item(), "meta-train acc": acc[0].item()}
                # )

        # assert 0
        return fast_weights, fast_negatives

    # def validate_pretrain(self, rank, net, val_loader, criterion, logs):
    #     net.eval()

    #     total_loss = 0
    #     with torch.no_grad():
    #         for batch_idx, data in enumerate(val_loader):
    #             features = data[1].cuda()
    #             pos_features = data[2].cuda()
    #             domains = data[3].cuda()

    #             if self.cfg.neg_per_domain:
    #                 all_logits = []
    #                 all_targets = []
    #                 for dom in self.all_domains:
    #                     dom_idx = torch.nonzero(domains == dom).squeeze()
    #                     if dom_idx.dim() == 0:
    #                         dom_idx = dom_idx.unsqueeze(0)
    #                     if torch.numel(dom_idx):
    #                         dom_features = features[dom_idx]
    #                         dom_pos_features = pos_features[dom_idx]
    #                         logits, targets = net(
    #                             dom_features, dom_pos_features, features.shape[0]
    #                         )
    #                         all_logits.append(logits)
    #                         all_targets.append(targets)
    #                 logits = torch.cat(all_logits, dim=0)
    #                 targets = torch.cat(all_targets, dim=0)
    #             else:
    #                 logits, targets = net(features, pos_features, features.shape[0])
    #             loss = criterion(logits, targets)
    #             total_loss += loss

    #         # if len(total_targets) > 0:
    #         #     total_targets = torch.cat(total_targets, dim=0)
    #         #     total_logits = torch.cat(total_logits, dim=0)
    #         #     acc1, acc5 = self.accuracy(total_logits, total_targets, topk=(1, 5))
    #         #     # f1, recall, precision = self.scores(total_logits, total_targets)
    #         total_loss /= len(val_loader)

    #         if rank == 0:
    #             log = f"[Pretrain] Validation Loss: {total_loss.item():.4f}"  # , Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
    #             logs.append(log)
    #             print(log)

    #         return total_loss.item()

    def finetune(
        self, rank, net, train_loader, criterion, optimizer, epoch, num_epochs, logs
    ):
        net.eval()

        for batch_idx, data in enumerate(train_loader):
            features = data[0].cuda()
            targets = data[3].cuda()
            targets = targets.type(torch.LongTensor).cuda()

            logits = net(features)
            loss = criterion(logits, targets)

            if rank == 0:
                if batch_idx % self.cfg.log_freq == 0:
                    acc1, acc5 = self.accuracy(logits, targets, topk=(1, 3))
                    log = f"Epoch [{epoch+1}/{num_epochs}]-({batch_idx}/{len(train_loader)}) "
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
