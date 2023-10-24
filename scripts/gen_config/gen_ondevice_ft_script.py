import os
import argparse
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretext', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=False, default=50)
    parser.add_argument('--batch_size', type=int, required=False, default=4)
    parser.add_argument('--shot', type=int, required=True)
    parser.add_argument('--seed', type=int, required=False, default=0)
    parser.add_argument('--domain_adaptation', action='store_true')
    parser.add_argument('--unfreeze', action='store_true')
    args = parser.parse_args()
    return args

DATA_PATH = '/home/hjyoon/data'
CONFIG_PATH = '/home/hjyoon/configs'
PRETRAINED_PATH = '/home/hjyoon/models'
NUM_CLS = 9

def run(args):
    pretext = args.pretext
    pretrained = f'{PRETRAINED_PATH}/{args.pretext}.pth.tar'
    config_path = f'{CONFIG_PATH}/{args.shot}shot/'
    data_path = f'{DATA_PATH}/{args.shot}shot'
    
    criterion = ''
    if args.pretext == 'cpc' or args.pretext == 'metacpc':
        criterion = 'crossentropy'
    
    config = f'''### On-device finetuning config
mode: finetune
seed: {args.seed}
num_workers: 1
episodes: 1

dataset_name: ichar
train_dataset_path: {data_path}/train.pkl
test_dataset_path: {data_path}/test.pkl
val_dataset_path: {data_path}/val.pkl
input_channels: 3
num_cls: {NUM_CLS}

optimizer: adam
criterion: {criterion}
start_epoch: 0
epochs: {args.epochs}
batch_size: {args.batch_size}
lr: 0.001
wd: 0.0

resume: ''
pretrained: {pretrained}
ckpt_dir: '.'
log_freq: 10
save_freq: -1

domain_adaptation: {'true' if args.domain_adaptation else 'false'}
task_steps: 10
task_lr: 0.001
reg_lambda: 0
no_vars: true
mlp: false
freeze: {'false' if args.unfreeze else 'true'}

pretext: {pretext}
# FOR CPC
enc_blocks: 4
kernel_sizes: [8, 4, 2, 1]
strides: [4, 2, 1, 1]
agg_blocks: 5
z_dim: 256
pooling: mean
pred_steps: 12
n_negatives: 15
offset: 4
'''
    filename = f'{pretext}'
    file_path = os.path.join(config_path, f'{filename}.yaml')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # generate config file
    with open(file_path, 'w') as f:
        f.write(config)

if __name__ == '__main__':
    args = parse_args()
    run(args)