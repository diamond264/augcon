#!/usr/bin/env python

import os
from glob import glob
prefix = 'imagenet_resnet50_sgd_lr001_m09_wd1e4_ts128'
pretext = "metasimsiam"
folder = ""
dataset = "imagenet_mini"
if "meta" in pretext:
    folder = "reptile"
config_path = f'/home/jaehyun98/git/aaa/config/image_preliminary/{dataset}/pretrain/{folder}/{prefix}'

def run():
    gpu = 0
    # domains = ["mnist_m"]
    domains = [""]
    gpus = [[1], [2]]
    urls = ['tcp://localhost:10021', 'tcp://localhost:10023']
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
dataset_name: {dataset}
train_dataset_path: /mnt/sting/hjyoon/projects/cross/ImageNet_ILSVRC2012_mini/train
val_dataset_path: /mnt/sting/hjyoon/projects/cross/ImageNet_ILSVRC2012_mini/val
test_dataset_path: /mnt/sting/hjyoon/projects/cross/ImageNet_ILSVRC2012_mini/val
domains: ['{domain}']

### Training config
optimizer: sgd
criterion: crossentropy
start_epoch: 0
epochs: 1000
batch_size: 512
lr: 0.01
momentum: 0.9
wd: 0.0001
down_sample : false
rand_aug: false

### Logs and checkpoints
resume: ''
ckpt_dir: /mnt/sting/jaehyun/aaa/models/digit5/{folder}/{prefix}/pretrain_single_source/{domain}
pretrained: ''
log_freq: 10
save_freq: 100

### Model config
pretext: {pretext}
backbone: imagenet_resnet50
out_dim: 2048
pred_dim: 512
pretrain_mlp: true
finetune_mlp: false

epsilone_final: 0
epsilone_start: 1

task_size: 128
inner_steps: 10
inner_batch_size: 10
num_task: 0
multi_cond_num_task: 5
log_steps: True
epsilone: 0.1
meta_lr: 0.01'''
        if flag == 0:
            flag = 1
        elif flag == 1:
            flag = 0
        file_path = os.path.join(config_path, f'SimSiam_ImageNet_pretrain.yaml')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(config)


if __name__ == '__main__':
    run()