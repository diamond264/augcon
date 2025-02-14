import torch
import time
# import psutil
from resource_metrics.procps import procps_all

import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader


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
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, test_dataset):
        print("Executing Meta TPN")
        self.main_worker(test_dataset)

    def main_worker(self, test_dataset):
        # Model initialization
        net = TPNNet(self.cfg.input_channels, self.cfg.z_dim, self.cfg.out_dim)
        cls_net = TPNClassifier(
            self.cfg.input_channels, self.cfg.z_dim, self.cfg.num_cls, self.cfg.mlp
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=True,
        )
        meta_train_dataset = test_dataset

        # Define criterion
        if self.cfg.criterion == "crossentropy":
            criterion = nn.CrossEntropyLoss()

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

        # Meta-train the pretrained model for domain adaptation
        if self.cfg.domain_adaptation:
            net.eval()
            net.zero_grad()
            support = []
            target_support = []
            for e in meta_train_dataset:
                support.append(e[1])
                target_support.append(torch.tensor(e[2]))
            support = torch.stack(support, dim=0)
            target_support = torch.stack(target_support, dim=0)

            time.sleep(3)
            start_time = time.time()
            enc_parameters = self.meta_train(net, support, target_support, criterion)
            print(f"Domain Adaptation Time: {time.time() - start_time}")
        else:
            enc_parameters = list(net.parameters())

        enc_dict = {}
        for idx, k in enumerate(list(cls_net.state_dict().keys())):
            if not "classifier" in k:
                enc_dict[k] = enc_parameters[idx]

        msg = cls_net.load_state_dict(enc_dict, strict=False)

        time.sleep(3)
        start_time = time.time()
        for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
            self.finetune(
                cls_net, test_loader, criterion, optimizer, epoch, self.cfg.epochs
            )
        print(f"Finetune Time: {time.time() - start_time}")

    def meta_train(self, net, support, target_support, criterion):
        fast_weights = list(net.parameters())
        for i in range(self.cfg.task_steps):
            shuffled_idx = torch.randperm(len(support))
            support = support[shuffled_idx]
            target_support = target_support[shuffled_idx]
            s_logits = net(support, fast_weights)
            s_loss = 0

            for j in range(len(s_logits)):
                s_loss += criterion(s_logits[j], target_support[:, j])
            s_loss /= len(s_logits)

            grad = torch.autograd.grad(s_loss, fast_weights)
            fast_weights = list(
                map(lambda p: p[1] - self.cfg.task_lr * p[0], zip(grad, fast_weights))
            )

            prefix = "Domain Adaptation"
        
            # cpu_usage = psutil.cpu_percent()  # CPU usage in %
            # ram_usage = psutil.virtual_memory().used / 1e6  # RAM usage in MB

            cpu_usage, ram_usage = procps_all()


            print(f"{prefix} [{i}/{self.cfg.task_steps}]")
            print(f"{prefix} CPU Usage: {cpu_usage}%")
            print(f"{prefix} RAM Usage: {ram_usage:.2f} MB")

        return fast_weights

    def finetune(self, net, train_loader, criterion, optimizer, epoch, num_epochs):
        net.eval()

        for batch_idx, data in enumerate(train_loader):
            features = data[0]
            targets = data[3]
            targets = targets.type(torch.LongTensor)

            logits = net(features)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        prefix = "Finetune"
        
        # cpu_usage = psutil.cpu_percent()  # CPU usage in %
        # ram_usage = psutil.virtual_memory().used / 1e6  # RAM usage in MB

        cpu_usage, ram_usage = procps_all()


        print(f"{prefix} [{epoch+1}/{num_epochs}]")
        print(f"{prefix} CPU Usage: {cpu_usage}%")
        print(f"{prefix} RAM Usage: {ram_usage:.2f} MB")
