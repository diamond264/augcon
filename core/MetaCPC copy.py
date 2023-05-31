import torch
import torch.nn as nn
import torch.nn.functional as F

# reference:
# https://github.com/facebookresearch/fairseq/blob/176cd934982212a4f75e0669ee81b834ee71dbb0/fairseq/models/wav2vec/wav2vec.py#L431
class Encoder(nn.Module):
    def __init__(self, input_channels=3, z_dim=256):
        super().__init__()
        layers = [
            nn.Conv1d(input_channels, 32, kernel_size=8, stride=4),#, padding_mode="reflect"),
            # nn.Conv1d(input_channels, 32, kernel_size=8, stride=4),#, padding_mode="reflect"),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(32, 64, kernel_size=8, stride=4),
            # nn.Conv1d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(64, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(128, z_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        ]
        self.strides = [4, 4, 1, 1]
        
        self.vars = nn.ParameterList()
        idx = 0
        for i in range(4):
            # Conv1d
            w = nn.Parameter(torch.ones_like(layers[idx].weight))
            torch.nn.init.kaiming_normal_(w)
            b = nn.Parameter(torch.zeros_like(layers[idx].bias))
            self.vars.append(w)
            self.vars.append(b)
            idx += 3
        
    def forward(self, x, vars=None, training=False):
        if vars is None:
            vars = self.vars
        idx = 0
        
        for i in range(4):
            w, b = vars[idx], vars[idx+1]
            x = F.conv1d(x, w, b, self.strides[i])
            idx += 2
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
    def __init__(self, num_blocks=5, num_filters=256):
        super(Aggregator, self).__init__()
        self.num_blocks = num_blocks
        # define convolutional blocks with increasing kernel sizes
        kernel_sizes = [2, 3, 4, 5, 6, 7]
        blocks = []
        for i in range(num_blocks):
            k = kernel_sizes[i]
            zero_pad = LeftZeroPad1d(k-1)
            conv_layer = nn.Conv1d(num_filters, num_filters, kernel_size=k, dilation=1)
            norm_layer = nn.GroupNorm(1, num_filters)
            relu_layer = nn.ReLU()
            blocks.append([zero_pad, conv_layer, norm_layer, relu_layer])
            
        # define skip connections
        self.skip_connections = nn.ModuleList()
        for i in range(num_blocks):
            # skip = nn.Conv1d(256, 256, kernel_size=1)
            skip = None
            self.skip_connections.append(skip)
            
        self.vars = nn.ParameterList()
        self.zero_pads = []
        for i in range(self.num_blocks):
            # LeftZeroPad1d
            self.zero_pads.append(blocks[i][0])
            # Conv1d
            w = nn.Parameter(torch.ones_like(blocks[i][1].weight))
            torch.nn.init.kaiming_normal_(w)
            b = nn.Parameter(torch.zeros_like(blocks[i][1].bias))
            self.vars.append(w)
            self.vars.append(b)
            # GroupNorm
            w = nn.Parameter(torch.ones_like(blocks[i][2].weight))
            b = nn.Parameter(torch.zeros_like(blocks[i][2].bias))
            self.vars.append(w)
            self.vars.append(b)
            
    def forward(self, z, vars=None):
        if vars is None:
            vars = self.vars
        idx = 0
        
        for i in range(self.num_blocks):
            residual = z
            z = self.zero_pads[i](z)
            
            w, b = vars[idx], vars[idx+1]
            z = F.conv1d(z, w, b)
            idx += 2
            
            w, b = vars[idx], vars[idx+1]
            z = F.group_norm(z, 1, w, b)
            idx += 2
            
            z = F.relu(z)
            # z_block = z_block[:, :, :-self.blocks[i][0].padding[0]]
            # pass output of each block through skip connection
            if i < self.num_blocks-1:
                if self.skip_connections[i] != None:
                    z = z + self.skip_connections[i](residual)
                else: z = z + residual
        # concatenate output of all blocks and return
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

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)

class FuturePredictor(nn.Module):
    def __init__(self, encoder, aggregator, z_dim=256, pred_steps=12, n_negatives=15, offset=16):
        super(FuturePredictor, self).__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.z_dim = z_dim
        self.pred_steps = pred_steps
        self.n_negatives = n_negatives
        self.offset = offset
        predictor = self.build_predictor()
        
        self.vars = nn.ParameterList()
        self.enc_param_idx = len(self.encoder.parameters())
        self.agg_param_idx = len(self.encoder.parameters())+len(self.aggregator.parameters())
        # self.vars.extend(self.encoder.parameters())
        # self.vars.extend(self.aggregator.parameters())
        w = nn.Parameter(torch.ones_like(predictor.weight))
        torch.nn.init.kaiming_normal_(w)
        b = nn.Parameter(torch.zeros_like(predictor.bias))
        self.vars.append(w)
        self.vars.append(b)
    
    def build_gru_aggregator(self, z_dim):
        aggregator = nn.Sequential(
                    TransposeLast(),
                    nn.GRU(
                        input_size=z_dim,
                        hidden_size=z_dim,
                        num_layers=1
                    ),
                    TransposeLast(deconstruct_idx=0),
        )
        return aggregator
    
    def build_predictor(self):
        predictor = nn.ConvTranspose2d(
            self.z_dim, self.z_dim, (1, self.pred_steps)
        )
        return predictor

    def sample_negatives(self, y, neg_idxs=None):
        bsz, fsz, tsz = y.shape
        high = tsz
        cross_high = tsz * bsz

        y = y.transpose(0, 1)  # BCT -> CBT
        y = y.contiguous().view(fsz, -1)  # CBT => C(BxT)
        
        if neg_idxs != None:
            negs = y[..., neg_idxs.view(-1)]
            negs = negs.view(
                fsz, bsz, self.n_negatives, tsz
            ).permute(
                2, 1, 0, 3
            )  # to NxBxCxT
            return negs, neg_idxs

        # neg_idxs = torch.randint(low=0, high=tsz, size=(bsz, self.n_negatives * tsz))
        # neg_idxs = torch.randint(low=0, high=tsz, size=(bsz, 0 * tsz))

        with torch.no_grad():
            # # perform inside-sample sampling
            # if self.n_negatives > 0:
            #     tszs = (
            #         buffered_arange(tsz)
            #         .unsqueeze(-1)
            #         .expand(-1, self.n_negatives)
            #         .flatten()
            #     )

            #     neg_idxs = torch.randint(
            #         low=0, high=high - 1, size=(bsz, self.n_negatives * tsz)
            #     )
            #     neg_idxs[neg_idxs >= tszs] += 1
            
            # perform cross-sample sampling
            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(tsz)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.n_negatives * tsz),
                )
                
                for i, row in enumerate(cross_neg_idxs):
                    row[row >= tszs+i*bsz] += 1
                # cross_neg_idxs[cross_neg_idxs >= tszs] += 1
        neg_idxs = cross_neg_idxs
        negs = y[..., neg_idxs.view(-1)]
        negs = negs.view(
            fsz, bsz, self.n_negatives, tsz
        ).permute(
            2, 1, 0, 3
        )  # to NxBxCxT
        return negs, neg_idxs

    def sample_neg_idxs(self, y):
        bsz, fsz, tsz = y.shape
        high = tsz
        cross_high = tsz * bsz

        y = y.transpose(0, 1)  # BCT -> CBT
        y = y.contiguous().view(fsz, -1)  # CBT => C(BxT)

        with torch.no_grad():
            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(tsz)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.n_negatives * tsz),
                )
                
                for i, row in enumerate(cross_neg_idxs):
                    row[row >= tszs+i*bsz] += 1
        neg_idxs = cross_neg_idxs
        return neg_idxs
            
    def forward(self, x, neg_samples, vars=None, neg_idxs=None):
        if vars is None:
            vars = nn.ParameterList()
            vars.extend(self.encoder.parameters())
            vars.extend(self.aggregator.parameters())
            vars.extend(self.vars)
        idx = 0
        
        enc_vars = vars[:self.enc_param_idx]
        z = self.encoder(x, enc_vars)
        neg_z = self.encoder(neg_samples, enc_vars)
        agg_vars = vars[self.enc_param_idx:self.agg_param_idx]
        z_hat = self.aggregator(z, agg_vars)
        z_pred = z_hat.unsqueeze(-1)
        w, b = vars[self.agg_param_idx], vars[self.agg_param_idx+1]
        z_pred = F.conv_transpose2d(z_pred, w, b)
        # z_pred = F.dropout(z_pred, 0.2)

        # negatives = self.sample_negatives(z)
        negatives, neg_idxs_ = self.sample_negatives(neg_z, neg_idxs)
        negatives = negatives[:,:90,:,:]
        z = z.unsqueeze(0)
        targets = torch.cat([z, negatives], dim=0)  # Copies x B x C x T

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

        predictions = predictions.view(-1, copies)*5

        return predictions, labels, neg_idxs_
    
    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                vars = nn.ParameterList()
                vars.extend(self.encoder.parameters())
                vars.extend(self.aggregator.parameters())
                vars.extend(self.vars)
                # for p in self.vars+self.encoder.vars+self.aggregator.vars:
                #     if p.grad is not None:
                #         p.grad.zero_()
            for p in vars:
                if p.grad is not None:
                    p.grad.zero_()

    def parameters(self):
        vars = nn.ParameterList()
        vars.extend(self.encoder.parameters())
        vars.extend(self.aggregator.parameters())
        vars.extend(self.vars)
        return vars


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