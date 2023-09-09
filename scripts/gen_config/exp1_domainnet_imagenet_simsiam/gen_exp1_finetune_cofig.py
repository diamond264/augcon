#!/usr/bin/env python

import os
from glob import glob

prefix = 'imagenet_finetune'
config_path = f'/home/jaehyun98/git/aaa/config/image_preliminary/domainnet/finetune/{prefix}'
learning_rate = 0.01
epochs = 50

def run():
    gpu = 4
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    for source in domains:
        config = f'''### Default config
mode: finetune
seed: 0
gpu: [{gpu}]
num_workers: 8
dist_url: tcp://localhost:10013

### Dataset config
dtype: 2d
dataset_name: pacs
train_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/finetune
val_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
test_dataset_path: /mnt/sting/hjyoon/projects/aaa/domainnet/data/test
domains: ["{source}"]
n_way: 5
k_shot: 5

### Training config
episodes: 1
optimizer: adam
criterion: crossentropy
start_epoch: 0
epochs: {epochs}
batch_size: 5
lr: {learning_rate}
momentum: 0.9
wd: 0.000
domain_adaptation: false
down_sample : false
rand_aug : false
supervised_adaptation : false
adapter : false

### Logs and checkpoints
resume: ''
# pretrained: clip
pretrained: /mnt/sting/hjyoon/projects/aaa/models/pre-trained/resnet50_imagenet/checkpoint_0099.pth.tar
ckpt_dir: /mnt/sting/jaehyun/aaa/models/domainnet/{prefix}/finetune_single_source/{source}
log_freq: 2
save_freq: 10
log_steps: true

### Model config
pretext: simsiam
backbone: imagenet_resenet50
out_dim: 1024
pred_dim: 512
pretrain_mlp: true
finetune_mlp: false
freeze: true

inner_steps: 10
inner_batch_size: 128
meta_lr: 0.00001
epsilone: 0.1'''
        gpu += 1
        if gpu >= 8: gpu = 4
        file_path = os.path.join(config_path, f'SimSiam_Domainnet_finetune_single_source_{source}.yaml')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(config)


if __name__ == '__main__':
    run()