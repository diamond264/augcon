import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# reference:
# https://github.com/facebookresearch/fairseq/blob/176cd934982212a4f75e0669ee81b834ee71dbb0/fairseq/models/wav2vec/wav2vec.py#L431
class Encoder(nn.Module):
    def __init__(self, input_channels=3, z_dim=256, num_blocks=4):
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks
        
        self.vars = nn.ParameterList()
        
        filters = [32, 64, 128, z_dim]
        self.kernel_sizes = [4, 2, 1, 1]
        self.strides = [2, 1, 1, 1]
        
        for i in range(num_blocks):
            block = nn.Sequential(nn.Conv1d(input_channels, filters[i],
                                            kernel_size=self.kernel_sizes[i],
                                            stride=self.strides[i]), 
                                  nn.ReLU(), 
                                  nn.Dropout(p=0.2))
            input_channels = filters[i]

            w = nn.Parameter(torch.ones_like(block[0].weight))
            torch.nn.init.kaiming_normal_(w)
            b = nn.Parameter(torch.zeros_like(block[0].bias))
            self.vars.append(w)
            self.vars.append(b)   
            
    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        
        idx = 0
        for i in range(self.num_blocks):
            w, b = vars[idx], vars[idx+1]
            idx += 2
            x = F.conv1d(x, w, b, self.strides[i])
            x = F.relu(x, True)
            x = F.dropout(x, 0.2)
            
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


class LeftZeroPad1d(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        
    def forward(self, x):
        # Compute the padding for the left side only
        padding = [self.k, 0]  # left, right
        x = F.pad(x, padding, "constant", 0)
        return x


class Aggregator(nn.Module):
    def __init__(self, num_blocks=5, num_filters=256, residual_scale=0.5):
        super(Aggregator, self).__init__()
        self.num_blocks = num_blocks
        
        self.vars = nn.ParameterList()
        self.zero_pads = []
        
        # define convolutional blocks with increasing kernel sizes
        kernel_sizes = [2, 3, 4, 5, 6, 7]
        for i in range(num_blocks):
            k = kernel_sizes[i]
            block = nn.Sequential(LeftZeroPad1d(k-1), 
                                  nn.Conv1d(num_filters, num_filters, kernel_size=k, dilation=1), 
                                  nn.GroupNorm(1, num_filters),
                                  nn.ReLU())
            # LeftZeroPad1d
            self.zero_pads.append(block[0])
            # Conv1d
            w = nn.Parameter(torch.ones_like(block[1].weight))
            torch.nn.init.kaiming_normal_(w)
            b = nn.Parameter(torch.zeros_like(block[1].bias))
            self.vars.append(w)
            self.vars.append(b)
            # GroupNorm
            w = nn.Parameter(torch.ones_like(block[2].weight))
            b = nn.Parameter(torch.zeros_like(block[2].bias))
            self.vars.append(w)
            self.vars.append(b)
            
        # define skip connections
        self.rproj = nn.ModuleList()
        for i in range(num_blocks):
            # rproj = nn.Conv1d(256, 256, kernel_size=1)
            rproj = None
            self.rproj.append(rproj)
        
        self.residual_scale = math.sqrt(residual_scale)
            
    def forward(self, z, vars=None):
        if vars is None:
            vars = self.vars
        
        idx = 0
        for i in range(self.num_blocks):
            residual = z
            z = self.zero_pads[i](z)
            
            w, b = vars[idx], vars[idx+1]
            idx += 2
            z = F.conv1d(z, w, b)
            
            w, b = vars[idx], vars[idx+1]
            idx += 2
            z = F.group_norm(z, 1, w, b)
            
            z = F.relu(z, True)
            
            rproj = self.rproj[i]
            if rproj != None:
                residual = rproj(residual)
            z = (z + residual) * self.residual_scale
        
        return z
    
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


class Predictor(nn.Module):
    def __init__(self, z_dim=256, pred_steps=12):
        super().__init__()
        self.vars = nn.ParameterList()
        
        self.pred_steps = pred_steps
        predictor = nn.ConvTranspose2d(
            z_dim, z_dim, (1, pred_steps)
        )
        w = nn.Parameter(torch.ones_like(predictor.weight))
        torch.nn.init.kaiming_normal_(w)
        b = nn.Parameter(torch.zeros_like(predictor.bias))
        self.vars.append(w)
        self.vars.append(b)
    
    def forward(self, z, vars=None):
        if vars is None:
            vars = self.vars
        
        z = F.conv_transpose2d(z, vars[0], vars[1])
        return z

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


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]

class CPCNet(nn.Module):
    def __init__(self, input_channels=3, z_dim=256, agg_blocks=4, pred_steps=12, n_negatives=15, offset=16):
        super(CPCNet, self).__init__()
        self.encoder = Encoder(input_channels, z_dim)
        self.aggregator = Aggregator(agg_blocks, z_dim)
        self.predictor = Predictor(z_dim, pred_steps)
        
        self.z_dim = z_dim
        self.pred_steps = pred_steps
        self.n_negatives = n_negatives
        self.offset = offset
        
        self.enc_param_idx = len(self.encoder.parameters())
        self.agg_param_idx = self.enc_param_idx+len(self.aggregator.parameters())
        self.pred_param_idx = self.agg_param_idx+len(self.predictor.parameters())

    def sample_negatives(self, y, n_negatives, bsz):
        _, fsz, tsz = y.shape
        cross_high = tsz * _

        y = y.transpose(0, 1)  # BCT -> CBT
        y = y.contiguous().view(fsz, -1)  # CBT => C(BxT)

        with torch.no_grad():
            # only perform cross-sample sampling
            if n_negatives > 0:
                tszs = (
                    buffered_arange(tsz)
                    .unsqueeze(-1)
                    .expand(-1, n_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, n_negatives * tsz),
                )
                for i, row in enumerate(cross_neg_idxs):
                    row[row >= tszs+i*bsz] += 1
                # cross_neg_idxs[cross_neg_idxs >= tszs] += 1
        neg_idxs = cross_neg_idxs

        negs = y[..., neg_idxs.view(-1)]
        negs = negs.view(
            fsz, bsz, n_negatives, tsz
        ).permute(
            2, 1, 0, 3
        )  # to NxBxCxT
        return negs
            
    def forward(self, x, vars=None, neg_x=None):
        if vars is None:
            vars = nn.ParameterList()
            vars.extend(self.encoder.parameters())
            vars.extend(self.aggregator.parameters())
            vars.extend(self.predictor.parameters())
            
        enc_vars = vars[:self.enc_param_idx]
        z = self.encoder(x, enc_vars)
        
        agg_vars = vars[self.enc_param_idx:self.agg_param_idx]
        z_hat = self.aggregator(z, agg_vars)
        z_hat = z_hat.unsqueeze(-1)
        
        pred_vars = vars[self.agg_param_idx:self.pred_param_idx]
        z_pred = self.predictor(z_hat, pred_vars)
        
        targets = z.unsqueeze(0)
        bsz = z.shape[0]
        if neg_x:
            neg_z = self.encoder(neg_x, enc_vars)
            neg_targets = self.sample_negatives(neg_z, self.n_negatives, bsz)
            targets = torch.cat([targets, neg_targets], dim=0)
        else:
            neg_targets = self.sample_negatives(z, self.n_negatives, bsz)
            targets = torch.cat([targets, neg_targets], dim=0)

        copies = targets.size(0)
        bsz, dim, tsz, steps = z_pred.shape
        steps = min(steps, tsz - self.offset)

        predictions = z_pred.new(
            bsz * copies * (tsz - self.offset + 1) * steps
            - ((steps + 1) * steps // 2) * copies * bsz
        )

        labels = predictions.new_full(
            (predictions.shape[0] // copies,), 0, dtype=torch.long
        )

        start = end = 0
        for i in range(steps):
            offset = i + self.offset
            end = start + (tsz - offset) * bsz * copies

            predictions[start:end] = torch.einsum(
                "bct,nbct->tbn", z_pred[..., :-offset, i], targets[..., offset:]
            ).flatten()

            start = end
        assert end == predictions.numel(), "{} != {}".format(end, predictions.numel())

        temp = 1
        predictions = predictions.view(-1, copies)/temp

        return predictions, labels
    
    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                vars = nn.ParameterList()
                vars.extend(self.encoder.parameters())
                vars.extend(self.aggregator.parameters())
                vars.extend(self.predictor.parameters())
            for p in vars:
                if p.grad is not None:
                    p.grad.zero_()

    def parameters(self):
        vars = nn.ParameterList()
        vars.extend(self.encoder.parameters())
        vars.extend(self.aggregator.parameters())
        vars.extend(self.predictor.parameters())
        return vars


class CPCClassifier(nn.Module):
    def __init__(self, encoder, aggregator, z_dim=256, seq_len=30, num_cls=6):
        super(CPCClassifier, self).__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.num_cls = num_cls
        self.fc = self.build_classifier()
    
    def build_classifier(self):
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.z_dim*self.seq_len, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, self.num_cls)
        )
        return classifier
            
    def forward(self, x):
        z = self.encoder(x)
        c = self.aggregator(z)
        pred = self.fc(c)
        return pred


class MetaCPCLearner:
    def __init__(self, cfg, gpu, logger):
        self.cfg = cfg
        self.gpu = gpu
        self.logger = logger
        
        self.net = CPCNet(cfg.input_channels, cfg.z_dim, cfg.agg_blocks,
                          cfg.pred_steps, cfg.n_negatives, cfg.offset)
        if len(gpu) > 1:
            self.net = nn.DataParallel(self.net, device_ids=gpu)
            torch.cuda.set_device(gpu[0])
        self.net = self.net.cuda()
        
        self.all_domains = cfg.domains
    
    def perform_train(self, train_loader, val_loader, test_loader, criterion, optimizer):
        for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
            self.train(train_loader, criterion, optimizer, epoch, self.cfg.epochs)
            self.validate(val_loader, criterion)
            
            ckpt_dir = self.cfg.ckpt_dir
            ckpt_filename = 'checkpoint_{:04d}.pth.tar'.format(epoch)
            ckpt_filename = os.path.join(ckpt_dir, ckpt_filename)
            state_dict = self.net.state_dict()
            self.save_checkpoint(ckpt_filename, epoch, state_dict, optimizer)
        
        self.validate(test_loader, criterion)
    
    def train(self, train_loader, criterion, optimizer, epoch, num_epochs):
        # switch to train mode
        self.net.train()
        
        for batch_idx, data in enumerate(train_loader):
            features = data[0].cuda()
            domains = data[2].cuda()
            
            if self.cfg.neg_per_domain:
                all_logits = []
                all_targets = []
                for dom in self.all_domains:
                    dom_idx = torch.nonzero(domains == dom).squeeze()
                    if dom_idx.dim() == 0: dom_idx = dom_idx.unsqueeze(0)
                    if torch.numel(dom_idx):
                        dom_features = features[dom_idx]
                        logits, targets = self.net(dom_features)
                        all_logits.append(logits)
                        all_targets.append(targets)
                logits = torch.cat(all_logits, dim=0)
                targets = torch.cat(all_targets, dim=0)
            else:
                logits, targets = self.net(features)
                
            loss = criterion(logits, targets)

            if batch_idx % self.cfg.log_freq == 0:
                acc1, acc5 = self.accuracy(logits, targets, topk=(1, 5))
                log = f'Epoch [{epoch+1}/{num_epochs}]-({batch_idx}/{len(train_loader)}) '
                log += f'Loss: {loss.item():.4f}, Acc(1): {acc1.item():.2f}, Acc(5): {acc5.item():.2f}'
                self.logger.info(log)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def validate(self, val_loader, criterion):
        pass
    
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

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)
    
    def parameters(self):
        return list(filter(lambda p: p.requires_grad, self.net.parameters()))


class Classifier(nn.Module):
    def __init__(self, encoder, aggregator, z_dim=256, seq_len=30, num_cls=6):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.num_cls = num_cls
        self.fc = self.build_classifier()
    
    def build_classifier(self):
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.z_dim*self.seq_len, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, self.num_cls)
        )
        return classifier
            
    def forward(self, x):
        # print(x.shape)
        z = self.encoder(x)
        # print(z.shape)
        c = self.aggregator(z)
        # print(c.shape)
        # assert(0)
        pred = self.fc(c)
        return pred