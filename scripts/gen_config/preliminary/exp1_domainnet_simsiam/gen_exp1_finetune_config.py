#!/usr/bin/env python

import os
from glob import glob

config_path = '/home/hjyoon/projects/augcon/config/image_preliminary/finetune'

def run():
    gpu = 0
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    meta_epoch = '0399'
    epoch = '0039'
    lr = '0.003'
    for source in domains:
        for target in domains:
            config = f'''### Default config
mode: finetune
seed: 0
gpu: [{gpu}]
num_workers: 2
dist_url: tcp://localhost:10002

### Dataset config
dtype: 2d
dataset_name: domainnet
train_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/finetune
val_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
test_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
domains: ["{target}"]
# ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
n_way: 5
k_shot: 10

### Training config
episodes: 1
optimizer: sgd
criterion: crossentropy
start_epoch: 0
epochs: 50
batch_size: 4
lr: {lr}
momentum: 0.9
wd: 0.0001
domain_adaptation: true
supervised_adaptation: false

### Logs and checkpoints
resume: ''
pretrained: /mnt/sting/hjyoon/projects/aaa/models/domainnet/pretrain_meta_except/{source}/checkpoint_{meta_epoch}.pth.tar
ckpt_dir: ./temp/finetune
log_freq: 20
save_freq: 40
log_steps: true

### Model config
pretext: metasimsiam
backbone: resnet18
out_dim: 2048
pred_dim: 512
pretrain_mlp: true
finetune_mlp: false
freeze: true

rand_aug: false
inner_steps: 10
inner_batch_size: 128
meta_lr: 0.00005
epsilone: 0.1'''
            # gpu += 1
            # if gpu >= 8: gpu = 0
            file_path = os.path.join(config_path, f'pt_meta_wo_{source}', f'SimSiam_DomainNet_ft_{target}.yaml')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(config)
            
            config = f'''### Default config
mode: finetune
seed: 0
gpu: [{gpu}]
num_workers: 8
dist_url: tcp://localhost:10002

### Dataset config
dtype: 2d
dataset_name: domainnet
train_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/finetune
val_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
test_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
domains: ["{target}"]
# ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
n_way: 5
k_shot: 10

### Training config
episodes: 1
optimizer: sgd
criterion: crossentropy
start_epoch: 0
epochs: 50
batch_size: 4
lr: {lr}
momentum: 0.9
wd: 0.0001
domain_adaptation: false
supervised_adaptation: false

### Logs and checkpoints
resume: ''
# pretrained: /mnt/sting/hjyoon/projects/aaa/models/domainnet/pretrain_except/{source}/checkpoint_{epoch}.pth.tar
pretrained: /mnt/sting/hjyoon/projects/aaa/models/domainnet/pretrain_except/{source}_05/checkpoint_{epoch}.pth.tar
# pretrained: ''
ckpt_dir: ./temp/finetune
log_freq: 20
save_freq: 40
log_steps: true

### Model config
pretext: metasimsiam
backbone: resnet18
out_dim: 2048
pred_dim: 512
pretrain_mlp: true
finetune_mlp: false
freeze: true

rand_aug: false
inner_steps: 10
inner_batch_size: 128
meta_lr: 0.00001
epsilone: 0.1'''
            # gpu += 1
            # if gpu >= 8: gpu = 0
            file_path = os.path.join(config_path, f'pt_wo_{source}', f'SimSiam_DomainNet_ft_{target}.yaml')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(config)
            
            config = f'''### Default config
mode: finetune
seed: 0
gpu: [{gpu}]
num_workers: 8
dist_url: tcp://localhost:10002

### Dataset config
dtype: 2d
dataset_name: domainnet
train_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/finetune
val_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
test_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
domains: ["{target}"]
# ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
n_way: 5
k_shot: 10

### Training config
episodes: 1
optimizer: sgd
criterion: crossentropy
start_epoch: 0
epochs: 50
batch_size: 4
lr: {lr}
momentum: 0.9
wd: 0.0001
domain_adaptation: false
supervised_adaptation: false

### Logs and checkpoints
resume: ''
# pretrained: /mnt/sting/hjyoon/projects/aaa/models/domainnet/pretrain_single_source/{source}2/checkpoint_{epoch}.pth.tar
pretrained: /mnt/sting/hjyoon/projects/aaa/models/domainnet/pretrain_single_source/{source}_05/checkpoint_{epoch}.pth.tar
# pretrained: ''
ckpt_dir: ./temp/finetune
log_freq: 20
save_freq: 40
log_steps: true

### Model config
pretext: metasimsiam
backbone: resnet18
out_dim: 2048
pred_dim: 512
pretrain_mlp: true
finetune_mlp: false
freeze: true

rand_aug: false
inner_steps: 10
inner_batch_size: 128
meta_lr: 0.00001
epsilone: 0.1'''
            gpu += 1
            if gpu >= 8: gpu = 0
            file_path = os.path.join(config_path, f'pt_w_{source}', f'SimSiam_DomainNet_ft_{target}.yaml')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(config)

if __name__ == '__main__':
    run()