import torch
import torch.nn as nn
import torch.nn.functional as F

# reference:
# https://github.com/facebookresearch/fairseq/blob/176cd934982212a4f75e0669ee81b834ee71dbb0/fairseq/models/wav2vec/wav2vec.py#L431
class CPCEncoder(nn.Module):
    def __init__(self, input_channels=3, z_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=4, stride=2),#, padding_mode="reflect"),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(32, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(64, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(128, z_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        return z


class LeftZeroPad1d(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        
    def forward(self, x):
        # Compute the padding for the left side only
        padding = [self.k, 0]  # left, right
        x = F.pad(x, padding, "constant", 0)
        return x


class CPCAggregator(nn.Module):
    def __init__(self, num_blocks=5, num_filters=256):
        super(CPCAggregator, self).__init__()
        self.num_blocks = num_blocks
        # define convolutional blocks with increasing kernel sizes
        kernel_sizes = [2, 3, 4, 5, 6, 7]
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            k = kernel_sizes[i]
            zero_pad = LeftZeroPad1d(k-1)
            conv_layer = nn.Conv1d(num_filters, num_filters, kernel_size=k, dilation=1)
            norm_layer = nn.GroupNorm(1, num_filters)
            relu_layer = nn.ReLU()
            self.blocks.append(nn.Sequential(zero_pad, conv_layer, norm_layer, relu_layer))
            
        # define skip connections
        self.skip_connections = nn.ModuleList()
        for i in range(num_blocks):
            # skip = nn.Conv1d(256, 256, kernel_size=1)
            skip = None
            self.skip_connections.append(skip)
            
    def forward(self, z):
        # perform autoregressive summarization with convolutional blocks
        block_outputs = []
        for i in range(self.num_blocks):
            residual = z
            for layer in self.blocks[i]:
                z = layer(z)
            # z_block = z_block[:, :, :-self.blocks[i][0].padding[0]]
            # pass output of each block through skip connection
            if i < self.num_blocks-1:
                if self.skip_connections[i] != None:
                    z = z + self.skip_connections[i](residual)
                else: z = z + residual
        # concatenate output of all blocks and return
        return z


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

class CPCFuturePredictor(nn.Module):
    def __init__(self, encoder, aggregator, z_dim=256, pred_steps=12, n_negatives=15, offset=16):
        super(CPCFuturePredictor, self).__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        # self.aggregator = nn.Sequential(
        #             TransposeLast(),
        #             nn.GRU(
        #                 input_size=z_dim,
        #                 hidden_size=z_dim,
        #                 num_layers=1
        #             ),
        #             TransposeLast(deconstruct_idx=0),
        #         )
        self.z_dim = z_dim
        self.pred_steps = pred_steps
        self.n_negatives = n_negatives
        self.offset = offset
        self.predictor = self.build_predictor()
    
    def build_predictor(self):
        predictor = nn.ConvTranspose2d(
            self.z_dim, self.z_dim, (1, self.pred_steps)
        )
        return predictor

    def sample_negatives(self, y):
        bsz, fsz, tsz = y.shape
        cross_high = tsz * bsz

        y = y.transpose(0, 1)  # BCT -> CBT
        y = y.contiguous().view(fsz, -1)  # CBT => C(BxT)

        # neg_idxs = torch.randint(low=0, high=tsz, size=(bsz, self.n_negatives * tsz))
        neg_idxs = torch.randint(low=0, high=tsz, size=(bsz, 0 * tsz))

        with torch.no_grad():
            # only perform cross-sample sampling
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
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1
        neg_idxs = cross_neg_idxs

        negs = y[..., neg_idxs.view(-1)]
        negs = negs.view(
            fsz, bsz, self.n_negatives, tsz
        ).permute(
            2, 1, 0, 3
        )  # to NxBxCxT
        return negs
            
    def forward(self, x):
        z = self.encoder(x)
        z_hat = self.aggregator(z)
        z_pred = z_hat.unsqueeze(-1)
        z_pred = self.predictor(z_pred)

        negatives = self.sample_negatives(z)
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

        predictions = predictions.view(-1, copies)

        return predictions, labels
