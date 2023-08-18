#!/usr/bin/env python

import os
from glob import glob

config_path = '/home/hjyoon/projects/augcon/config/image_preliminary/finetune'

def run():
    gpu = 0
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    for source in domains+["none"]:
        for target in domains:
            config = f'''### Default config
mode: finetune
seed: 0
gpu: [{gpu}]
num_workers: 8
dist_url: tcp://localhost:10002

### Dataset config
dtype: 2d
dataset_name: domainnet
train_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/finetune/10shot
val_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
test_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
domains: ["{target}"]
# ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
augmentations: ["blur"]
input_channels: 3
num_cls: 224

### Training config
optimizer: adam
criterion: crossentropy
start_epoch: 0
epochs: 40
batch_size: 64
lr: 0.005
momentum: 0.0
wd: 0.0001

### Logs and checkpoints
resume: ''
pretrained: ./temp/pretrain_except_{source}/checkpoint_0099.pth.tar
# pretrained: ''
ckpt_dir: ./temp/finetune
log_freq: 4
save_freq: 10

### Model config
pretext: simsiam
backbone: resnet18
mlp: false
freeze: true
no_vars: true'''
            gpu += 1
            if gpu >= 8: gpu = 0
            file_path = os.path.join(config_path, f'pt_wo_{source}', f'SimSiam_DomainNet_ft_{target}.yaml')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(config)

if __name__ == '__main__':
    run()