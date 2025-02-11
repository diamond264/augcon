import copy
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


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
    def __init__(self, input_channels, z_dim):
        super(Encoder, self).__init__()
        self.vars = nn.ParameterList()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

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

    def __init__(self, input_channels=3, z_dim=96, out_dim=50, T=0.1, mlp=True):
        super(SimCLRNet, self).__init__()
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder = Encoder(input_channels, z_dim)

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
        masks = F.one_hot(torch.arange(z.size(0)), z.size(0))

        pos_logits = torch.matmul(z, aug_z.t())
        neg_logits_1 = torch.matmul(z, z.t())
        neg_logits_1 = neg_logits_1 - masks * LARGE_NUM
        neg_logits_2 = torch.matmul(aug_z, aug_z.t())
        neg_logits_2 = neg_logits_2 - masks * LARGE_NUM

        logits = torch.cat([pos_logits, neg_logits_1, neg_logits_2], dim=1)
        # logits = torch.cat([pos_logits], dim=1)
        logits /= self.T
        labels = torch.arange(z.size(0))

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
        masks = F.one_hot(torch.arange(z.size(0)), z.size(0))

        pos_logits = torch.matmul(z, aug_z.t())
        logits = torch.cat([pos_logits], dim=1)
        logits /= self.T
        labels = torch.arange(z.size(0))

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
        masks = F.one_hot(torch.arange(z.size(0)), z.size(0))

        pos_logits = torch.matmul(z, aug_z.t())
        neg_logits_1 = torch.matmul(z, z.t())
        neg_logits_1 = neg_logits_1 - masks * LARGE_NUM
        neg_logits_2 = torch.matmul(aug_z, aug_z.t())
        neg_logits_2 = neg_logits_2 - masks * LARGE_NUM

        logits = torch.cat([pos_logits, neg_logits_1, neg_logits_2], dim=1)
        logits /= self.T
        labels = torch.arange(z.size(0))

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
        masks = F.one_hot(torch.arange(z.size(0)), z.size(0))

        pos_logits = torch.matmul(z, aug_z.t())
        # pos_logits = torch.einsum("ij,ij->i", z, aug_z)
        # pos_logits = torch.unsqueeze(pos_logits, dim=1)
        neg_logits_1 = torch.matmul(z, z.t())
        neg_logits_1 = neg_logits_1 - masks * LARGE_NUM
        neg_logits_2 = torch.matmul(aug_z, aug_z.t())
        neg_logits_2 = neg_logits_2 - masks * LARGE_NUM

        neg_logits_3 = torch.matmul(z, neg_z.t())

        logits = torch.cat(
            [pos_logits, neg_logits_1, neg_logits_2, neg_logits_3],
            # [pos_logits, neg_logits_3],
            dim=1,
        )
        logits /= self.T
        labels = torch.arange(z.size(0))

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
        masks = F.one_hot(torch.arange(z.size(0)), z.size(0))

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
        labels = torch.arange(z.size(0))

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
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, test_dataset):
        print("Executing MetaSimCLR")
        self.main_worker(test_dataset)

    def main_worker(self, test_dataset):
        # Model initialization
        net = SimCLRNet(
            self.cfg.input_channels, self.cfg.z_dim, self.cfg.out_dim, self.cfg.T, True
        )
        cls_net = SimCLRClassifier(
            self.cfg.input_channels, self.cfg.z_dim, self.cfg.num_cls, self.cfg.mlp
        )

        meta_train_dataset = test_dataset
        collate_fn = None
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
            drop_last=True,
        )

        memory_bank = Adversary_Negatives(self.cfg.bank_size, self.cfg.out_dim)
        memory_bank.init(net, meta_train_dataset)

        criterion = nn.CrossEntropyLoss()
        net.eval()
        enc_parameters = list(copy.deepcopy(net.parameters()))

        # perform domgin adaptation
        if self.cfg.domain_adaptation:
            support = [e[1] for e in meta_train_dataset]
            support = torch.stack(support, dim=0)

            pos_support = [e[2] for e in meta_train_dataset]
            pos_support = torch.stack(pos_support, dim=0)

            time.sleep(3)
            start_time = time.time()
            enc_parameters, _ = self.meta_train(
                net, support, pos_support, memory_bank, criterion
            )
            print(f"Domain Adaptation Time: {time.time() - start_time}")

        enc_dict = {}
        for idx, k in enumerate(list(cls_net.state_dict().keys())):
            if not "classifier" in k:
                enc_dict[k] = enc_parameters[idx]
            else:
                enc_dict[k] = cls_net.state_dict()[k]
        msg = cls_net.load_state_dict(enc_dict, strict=True)
        net = cls_net
        if self.cfg.freeze:
            for name, param in net.named_parameters():
                if not "classifier" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
        optimizer = None
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

        time.sleep(3)
        start_time = time.time()
        for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
            self.finetune(
                net,
                test_loader,
                criterion,
                optimizer,
                epoch,
                self.cfg.epochs,
            )
        print(f"Finetune Time: {time.time() - start_time}")

    def meta_train(self, net, support, pos_support, memory_bank, criterion):
        fast_weights = list(net.parameters())
        fast_negatives = list(memory_bank.parameters())

        for i in range(self.cfg.task_steps):
            batch_size = len(support)
            for j in range(len(support) // batch_size):
                support_batch = support[j * batch_size : (j + 1) * batch_size]
                pos_support_batch = pos_support[j * batch_size : (j + 1) * batch_size]
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

                else:
                    support_ = support_batch
                    pos_support_ = pos_support_batch
                    s_logits, s_targets = net.forward(
                        support_, pos_support_, fast_weights
                    )
                    s_loss = criterion(s_logits, s_targets)

                    grad = torch.autograd.grad(s_loss, fast_weights)
                    fast_weights = list(
                        map(
                            lambda p: p[1] - self.cfg.task_lr * p[0],
                            zip(grad, fast_weights),
                        )
                    )

            prefix = "Domain Adaptation"
            cpu_usage = psutil.cpu_percent()  # CPU usage in %
            ram_usage = psutil.virtual_memory().used / 1e6  # RAM usage in MB
            print(f"{prefix} [{i}/{self.cfg.task_steps}]")
            print(f"{prefix} CPU Usage: {cpu_usage}%")
            print(f"{prefix} RAM Usage: {ram_usage:.2f} MB")

        return fast_weights, fast_negatives

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
        cpu_usage = psutil.cpu_percent()  # CPU usage in %
        ram_usage = psutil.virtual_memory().used / 1e6  # RAM usage in MB
        print(f"{prefix} [{epoch+1}/{num_epochs}]")
        print(f"{prefix} CPU Usage: {cpu_usage}%")
        print(f"{prefix} RAM Usage: {ram_usage:.2f} MB")
