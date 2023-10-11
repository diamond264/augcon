import os
import argparse
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=False, default=100)
    parser.add_argument('--batch_size', type=int, required=False, default=512)
    # recommended batch_size - 512 for target-only setting
    parser.add_argument('--target_only', action='store_true')
    parser.add_argument('--perdomain', action='store_true')
    parser.add_argument('--port', type=int, required=False, default=10001)
    parser.add_argument('--num_gpus', type=int, required=False, default=1)

    parser.add_argument('--lr', type=float, required=False, default=0.001)
    parser.add_argument('--wd', type=float, required=False, default=0.0001)
    parser.add_argument('--momentum', type=float, required=False, default=0)
    parser.add_argument('--prefix', type=str, required=False, default='')
    args = parser.parse_args()
    return args


data_paths = {'ichar': '/mnt/sting/hjyoon/projects/cross/ICHAR/augcon',
              'hhar': '/mnt/sting/hjyoon/projects/cross/HHAR/augcon',
              'opportunity': '/mnt/sting/hjyoon/projects/cross/Opportunity/augcon',
              'realworld': '/mnt/sting/hjyoon/projects/cross/RealWorld/augcon',
              'pamap2': '/mnt/sting/hjyoon/projects/cross/PAMAP2/augcon'}


def run(args):
    pretext = 'simclr'
    data_path = data_paths[args.dataset]
    specific_path = f'{args.prefix}lr{args.lr}_m{args.momentum}_wd{args.wd}'
    config_path = f'/mnt/sting/hjyoon/projects/aaa/configs/imwut/main/{args.dataset}/pretrain/{pretext}/{specific_path}'

    domains = glob(os.path.join(data_path, '*'))
    domains = [os.path.basename(domain) for domain in domains]
    print(domains)

    flag = 0
    for domain in domains:
        if args.num_gpus == 4:
            if flag == 0:
                gpu = [0, 1, 2, 3]
                dist_url = f'tcp://localhost:{args.port}'
                flag = 1
            elif flag == 1:
                gpu = [4, 5, 6, 7]
                dist_url = f'tcp://localhost:{args.port + 1}'
                flag = 0
        elif args.num_gpus == 1:
            gpu = [flag]
            dist_url = f'tcp://localhost:{args.port + flag}'
            flag += 1
            if flag == 4: flag = 0

        default_config = f'''### Default config
mode: pretrain
seed: 0 # fix as 0 in pretrain
gpu: {gpu}
num_workers: 8
dist_url: {dist_url}
'''
        dirname = 'pretrain_target' if args.target_only else 'pretrain'
        dataset_config = f'''### Dataset config
dataset_name: {args.dataset}
train_dataset_path: {data_path}/{domain}/{dirname}/train.pkl
test_dataset_path: {data_path}/{domain}/{dirname}/val.pkl
val_dataset_path: {data_path}/{domain}/{dirname}/val.pkl
input_channels: 3
num_cls: 9 # not important
'''
        training_config = f'''### Training config
optimizer: adam
criterion: crossentropy
start_epoch: 0
epochs: {args.epochs}
batch_size: {args.batch_size}
lr: {args.lr}
wd: {args.wd}
momentum : {args.momentum}
'''
        save_freq = args.epochs / 10
        ckpt_dir = f'/mnt/sting/hjyoon/projects/aaa/models/imwut/main/{args.dataset}/pretrain/{pretext}/without_{domain}/{specific_path}'
        if args.perdomain:
            ckpt_dir = f'/mnt/sting/hjyoon/projects/aaa/models/imwut/main/{args.dataset}/pretrain/{pretext}/perdomain_without_{domain}/{specific_path}'
        if args.target_only:
            ckpt_dir = ckpt_dir.replace('without', 'only')
        log_config = f'''### Logs and checkpoints
resume: ''
ckpt_dir: {ckpt_dir}
log_freq: 20
save_freq: {save_freq}
'''
        neg_per_domain = 'true' if args.perdomain else 'false'
        learning_config = f'neg_per_domain: {neg_per_domain}'
        model_config = f'''### Model config
pretext: {pretext}

#For simclr
out_dim: 50
T: 0.1
mlp: true
z_dim: 256

{learning_config}
'''
        config = f'''{default_config}
{dataset_config}
{training_config}
{log_config}
{model_config}
'''
        filename = f'without_{domain}'
        if args.target_only:
            filename = f'only_{domain}'
        if args.perdomain:
            filename = filename + '_perdomain'
        file_path = os.path.join(config_path, f'gpu{flag}_{filename}.yaml')
        # generate directory if not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # generate config file
        with open(file_path, 'w') as f:
            f.write(config)


if __name__ == '__main__':
    args = parse_args()
    run(args)