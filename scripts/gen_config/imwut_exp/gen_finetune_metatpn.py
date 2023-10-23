import os
import argparse
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=False, default=50)
    parser.add_argument('--batch_size', type=int, required=False, default=4)
    parser.add_argument('--shot', type=int, required=True)
    parser.add_argument('--seed', type=int, required=False, default=0)
    parser.add_argument('--port', type=int, required=False, default=10001)
    parser.add_argument('--num_gpus', type=int, required=False, default=1)
    parser.add_argument('--specific_path', type=str, required=False, default='')
    parser.add_argument('--perdomain', action='store_true')
    parser.add_argument('--target_only', action='store_true')
    parser.add_argument('--domain_adaptation', action='store_true')
    parser.add_argument('--task_lr', type=str, required=False, default=0.001)
    parser.add_argument('--task_step', type=int, required=False, default=10)
    args = parser.parse_args()
    return args


data_paths = {'ichar': '/mnt/sting/hjyoon/projects/cross/ICHAR/augcon',
              'hhar': '/mnt/sting/hjyoon/projects/cross/HHAR/augcon',
              'opportunity': '/mnt/sting/hjyoon/projects/cross/Opportunity/augcon',
              'realworld': '/mnt/sting/hjyoon/projects/cross/RealWorld/augcon',
              'pamap2': '/mnt/sting/hjyoon/projects/cross/PAMAP2/augcon'}

num_cls = {'ichar': 9,
           'hhar': 6,
           'opportunity': 4,
           'realworld': 8,
           'pamap2': 12}


def run(args):
    pretext = 'metatpn'
    data_path = data_paths[args.dataset]
    config_path = f'/mnt/sting/hjyoon/projects/aaa/configs/imwut/main/{args.dataset}/finetune_{args.shot}shot/{pretext}/{args.specific_path}/ts{args.task_step}_tlr{args.task_lr}'
    specific_path = args.specific_path

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
            if flag == 8: flag = 0

        default_config = f'''### Default config
mode: finetune
seed: {args.seed} # fix as 0 in pretrain
gpu: {gpu}
num_workers: 1
dist_url: {dist_url}
episodes: 1
'''
        dirname = f'finetune/{args.shot}shot/target'
        dataset_config = f'''### Dataset config
dataset_name: {args.dataset}
train_dataset_path: {data_path}/{domain}/{dirname}/train.pkl
test_dataset_path: {data_path}/{domain}/{dirname}/test.pkl
val_dataset_path: {data_path}/{domain}/{dirname}/val.pkl
input_channels: 3
num_cls: {num_cls[args.dataset]}
'''
        training_config = f'''### Training config
optimizer: adam
criterion: crossentropy
start_epoch: 0
epochs: {args.epochs}
batch_size: {args.batch_size}
lr: 0.001
wd: 0.0
'''
        save_freq = 10
        ckpt_dir = f'/mnt/sting/hjyoon/projects/aaa/models/imwut/main/{args.dataset}/finetune_{args.shot}shot/pretrained_simclr_'
        postfix = f'without'
        if args.target_only: postfix = f'only'
        if args.perdomain: postfix = 'perdomain_' + postfix
        pretrained = f'/mnt/sting/hjyoon/projects/aaa/models/imwut/main/{args.dataset}/pretrain/{pretext}/{postfix}_{domain}/{specific_path}/checkpoint_4999.pth.tar'
        if args.domain_adaptation:
            postfix = postfix + f'/da_true_seed_{args.seed}'
        else:
            postfix = postfix + f'/da_false_seed_{args.seed}'
        ckpt_dir = ckpt_dir + postfix
        log_config = f'''### Logs and checkpoints
resume: ''
pretrained: {pretrained}
ckpt_dir: {ckpt_dir}
log_freq: 5
save_freq: {save_freq}
'''
        learning_config = f'''### Meta-learning
domain_adaptation: {'true' if args.domain_adaptation else 'false'}
out_cls_neg_sampling: false

#For tpn
out_dim: 2
T: 0.1
z_dim: 96

task_steps: {args.task_step}
task_lr: {args.task_lr}
reg_lambda: 0
no_vars: true
mlp: false
freeze: true'''
        model_config = f'''### Model config
pretext: {pretext}


{learning_config}
'''
        config = f'''{default_config}
{dataset_config}
{training_config}
{log_config}
{model_config}
'''
        filename = f'{postfix}/gpu{flag}_{domain}'
        file_path = os.path.join(config_path, f'{filename}.yaml')
        # generate directory if not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # generate config file
        with open(file_path, 'w') as f:
            f.write(config)


if __name__ == '__main__':
    args = parse_args()
    run(args)