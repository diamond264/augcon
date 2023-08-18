#!/usr/bin/env python

import os
from glob import glob

config_path = '/home/hjyoon/projects/augcon/config/image_preliminary/pretrain'

def run():
    gpu = 0
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    gpus = [[0,1,2,3], [4,5,6,7]]
    urls = ['tcp://localhost:10001', 'tcp://localhost:10002']
    flag = 0
    for domain in domains:
        except_dom = domains.copy()
        except_dom.remove(domain)
        config = f'''### Default config
mode: pretrain
seed: 0
gpu: {gpus[flag]}
num_workers: 8
dist_url: {urls[flag]}

### Dataset config
dtype: 2d
dataset_name: domainnet
train_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/train
val_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
test_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
domains: {except_dom}
augmentations: ["blur"]
input_channels: 3
num_cls: 345

### Training config
optimizer: sgd
criterion: crossentropy
start_epoch: 0
epochs: 100
batch_size: 512
lr: 0.025
momentum: 0.9
wd: 0.0001

### Logs and checkpoints
# resume: ./temp/pretrain3/checkpoint_0099.pth.tar
resume: ''
ckpt_dir: ./temp/pretrain_except_{domain}
log_freq: 20
save_freq: 10

### Model config
pretext: simsiam
backbone: resnet18
out_dim: 2048
pred_dim: 512
mlp: true
freeze: false
neg_per_domain: false'''
        if flag == 0: flag = 1
        elif flag == 1: flag = 0
        file_path = os.path.join(config_path, f'SimSiam_DomainNet_pretrain_except_{domain}.yaml')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(config)
    
    config = f'''### Default config
mode: pretrain
seed: 0
gpu: {gpus[flag]}
num_workers: 8
dist_url: {urls[flag]}

### Dataset config
dtype: 2d
dataset_name: domainnet
train_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/train
val_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
test_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
domains: {domains}
augmentations: ["blur"]
input_channels: 3
num_cls: 345

### Training config
optimizer: sgd
criterion: crossentropy
start_epoch: 0
epochs: 100
batch_size: 512
lr: 0.025
momentum: 0.9
wd: 0.0001

### Logs and checkpoints
# resume: ./temp/pretrain3/checkpoint_0099.pth.tar
resume: ''
ckpt_dir: ./temp/pretrain_except_none
log_freq: 20
save_freq: 10

### Model config
pretext: simsiam
backbone: resnet18
out_dim: 2048
pred_dim: 512
mlp: true
freeze: false
neg_per_domain: false'''
    if flag == 0: flag = 1
    elif flag == 1: flag = 0
    file_path = os.path.join(config_path, f'SimSiam_DomainNet_pretrain_except_none.yaml')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(config)

if __name__ == '__main__':
    run()