#!/usr/bin/env python

import os
from glob import glob

data_path = '/mnt/sting/hjyoon/projects/cross/ICSR/augcon'
config_path = '/home/hjyoon/projects/augcon/config/exp0_icsr'

def pretty_print(domain_idx):
    for domain in domain_idx:
        print(f'{domain}: {domain_idx[domain]}')

def run():
    domains = glob(os.path.join(data_path, '*'))
    domains = [os.path.basename(domain) for domain in domains]
    
    domain_idx = {}
    gpu_idx = 0
    for domain in domains:
        domain_idx[domain] = gpu_idx
        gpu_idx += 1
        if gpu_idx == 8: gpu_idx = 0
    pretty_print(domain_idx)
    
    pretexts = ['cpc', 'metacpc']
    architecture = '8421'
    for seed in [0]:
        for domain_adaptation in ['true', 'false']:
            for pretext in pretexts:
                for src_domain in domains:
                    gpu = domain_idx[src_domain]
                    for tgt_domain in domains:
                        for perdomain in [True, False]:
                            default_config = f'''### Default config
mode: finetune
seed: {seed}
gpu: [{gpu}]
num_workers: 1
dist_url: tcp://localhost:10001
'''
                            dataset_config = f'''### Dataset config
dataset_name: icsr
train_dataset_path: /mnt/sting/hjyoon/projects/cross/ICSR/augcon/{tgt_domain}/finetune/10shot/target/train.pkl
test_dataset_path: /mnt/sting/hjyoon/projects/cross/ICSR/augcon/{tgt_domain}/finetune/10shot/target/test.pkl
val_dataset_path: /mnt/sting/hjyoon/projects/cross/ICSR/augcon/{tgt_domain}/finetune/10shot/target/val.pkl
input_channels: 1
num_cls: 14
'''
                            epochs = 50
                            training_config = f'''### Training config
optimizer: adam
criterion: crossentropy
start_epoch: 0
epochs: {epochs}
batch_size: 4
lr: 0.001
wd: 0.0
'''
                            pretrained_model = f'without_{src_domain}/{pretext}_perdomain_{architecture}' if perdomain else f'without_{src_domain}/{pretext}_random_{architecture}'
                            pretrained_ckpt = f'/checkpoint_0099.pth.tar' if pretext == 'cpc' else f'checkpoint_0999.pth.tar'
                            save_freq = 10
                            log_config = f'''### Logs and checkpoints
resume: ''
pretrained: /mnt/sting/hjyoon/projects/augcontrast/models/exp0_icsr/pretrain/{pretrained_model}/{pretrained_ckpt}
ckpt_dir: /mnt/sting/hjyoon/projects/augcontrast/models/exp0_icsr/finetune_da_{domain_adaptation}/{pretrained_model}/tgt_{tgt_domain}_seed{seed}
log_freq: 5
save_freq: {save_freq}
'''
                            if architecture == '8421':
                                kernel_sizes = '[16, 16, 8, 4]'
                                strides = '[8, 8, 4, 2]'
                            elif architecture == '4111':
                                kernel_sizes = '[4, 1, 1, 1]'
                                strides = '[2, 1, 1, 1]'
                            neg_per_domain = 'true' if perdomain else 'false'
                            learning_config = f'''### Meta-learning
domain_adaptation: {domain_adaptation}
task_steps: 10
task_lr: 0.0005
reg_lambda: 0
mlp: false
freeze: true'''
                            if pretext == 'cpc': learning_config += f'\nno_vars: true'
                            elif pretext == 'metacpc': learning_config += f'\nno_vars: false'
                            model_config = f'''### Model config
pretext: metacpc
## Encoder
enc_blocks: 4
kernel_sizes: {kernel_sizes}
strides: {strides}
## Aggregator
agg_blocks: 5
z_dim: 256
## Predictor
pooling: mean
pred_steps: 12
n_negatives: 15
offset: 4
{learning_config}
'''
                            config = f'''{default_config}
{dataset_config}
{training_config}
{log_config}
{model_config}
'''
                            perdom_type = 'perdomain' if perdomain else 'random'
                            file_path = os.path.join(config_path, f'finetune_da_{domain_adaptation}/{pretrained_model}/tgt_{tgt_domain}_seed{seed}.yaml')
                            # generate directory if not exist
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                            # generate config file
                            with open(file_path, 'w') as f:
                                f.write(config)

if __name__ == '__main__':
    run()