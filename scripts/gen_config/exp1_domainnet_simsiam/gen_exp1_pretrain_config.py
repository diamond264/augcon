#!/usr/bin/env python

import os
from glob import glob

config_path = '/home/hjyoon/projects/augcon/config/image_preliminary/pretrain'

def run():
    gpu = 0
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    gpus = [[0,1,2,3], [4,5,6,7]]
    urls = ['tcp://localhost:10001', 'tcp://localhost:10002']
    gpu = [0,1,2,3,4,5,6,7]
    url = 'tcp://localhost:10001'
    flag = 0
    for domain in domains:
        except_dom = domains.copy()
        except_dom.remove(domain)
        config = f'''### Default config
mode: pretrain
seed: 0
gpu: {gpu}
num_workers: 8
dist_url: {url}

### Dataset config
dtype: 2d
dataset_name: domainnet
train_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/pretrain
val_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
test_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
domains: ['{domain}']

### Training config
optimizer: sgd
criterion: crossentropy
start_epoch: 0
epochs: 100
batch_size: 512
lr: 0.001
momentum: 0.9
wd: 0.0001

### Logs and checkpoints
resume: ''
pretrained: clip
ckpt_dir: /mnt/sting/hjyoon/projects/aaa/models/simsiam_ClipInit_ResNet50_001/domainnet/pretrain_single_source/{domain}
log_freq: 50
save_freq: 10

### Model config
rand_aug: false
pretext: simsiam
backbone: resnet50
out_dim: 1024
pred_dim: 512
pretrain_mlp: true'''
        # if flag == 0: flag = 1
        # elif flag == 1: flag = 0
        file_path = os.path.join(config_path, f'SimSiam_DomainNet_pretrain_single_source_{domain}.yaml')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(config)
        
        config = f'''### Default config
mode: pretrain
seed: 0
gpu: {gpu}
num_workers: 8
dist_url: {url}

### Dataset config
dtype: 2d
dataset_name: domainnet
train_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/pretrain
val_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
test_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
domains: {except_dom}

### Training config
optimizer: sgd
criterion: crossentropy
start_epoch: 0
epochs: 100
batch_size: 512
lr: 0.001
momentum: 0.9
wd: 0.0001

### Logs and checkpoints
resume: ''
pretrained: clip
ckpt_dir: /mnt/sting/hjyoon/projects/aaa/models/simsiam_ClipInit_ResNet50_001/domainnet/pretrain_except/{domain}
log_freq: 50
save_freq: 10

### Model config
rand_aug: false
pretext: simsiam
backbone: resnet50
out_dim: 1024
pred_dim: 512
pretrain_mlp: true'''
        if flag == 0: flag = 1
        elif flag == 1: flag = 0
        file_path = os.path.join(config_path, f'SimSiam_DomainNet_pretrain_except_{domain}.yaml')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(config)

if __name__ == '__main__':
    run()