#!/usr/bin/env python

import os
from glob import glob

config_path = '/home/hjyoon/projects/augcon/config/image_preliminary/pretrain'

def run():
    gpu = 0
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    gpu = 0
    url = 'tcp://localhost:10001'
    for domain in domains:
        except_dom = domains.copy()
        except_dom.remove(domain)
        config = f'''### Default config
mode: pretrain
seed: 0
gpu: [{gpu}]
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
epochs: 500
batch_size: 64
lr: 0.1
momentum: 0.9
wd: 0.0001

### Logs and checkpoints
resume: ''
ckpt_dir: /mnt/sting/hjyoon/projects/aaa/models/domainnet/pretrain_meta_except/{domain}
log_freq: 50
save_freq: 100

### Model config
pretext: metasimsiam
backbone: resnet18
out_dim: 2048
pred_dim: 512
pretrain_mlp: true

rand_aug: true
task_size: 20
inner_steps: 10
inner_batch_size: 10
num_task: 5
multi_cond_num_task: 0
log_steps: True
epsilone: 0.15
meta_lr: 0.00005'''

        gpu += 1
        if gpu == 8: gpu = 0
        file_path = os.path.join(config_path, f'SimSiam_DomainNet_pretrain_meta_except_{domain}.yaml')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(config)
            

if __name__ == '__main__':
    run()