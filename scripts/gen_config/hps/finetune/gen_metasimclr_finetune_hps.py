import os
from glob import glob

PRETEXT = 'metasimclr'
PRETRAIN_CRITERION = 'crossentropy'
PRETRAIN_HPS = {
    'lr': [0.001],
    'epoch': ['4999']
}

DATASETS = ['ichar', 'hhar', 'pamap2', 'dsa']
DATA_PATH = {
    'ichar': ['/mnt/sting/hjyoon/projects/cross/ICHAR/augcon/target_domain_WA0002-bkkim/',
              '/mnt/sting/hjyoon/projects/cross/ICHAR/augcon/target_domain_PH0012-thanh/',
              '/mnt/sting/hjyoon/projects/cross/ICHAR/augcon/target_domain_PH0038-iygoo/'],
    'hhar': ['/mnt/sting/hjyoon/projects/cross/HHAR/augcon/target_user_a_model_nexus4/',
             '/mnt/sting/hjyoon/projects/cross/HHAR/augcon/target_user_c_model_s3/',
             '/mnt/sting/hjyoon/projects/cross/HHAR/augcon/target_user_f_model_lgwatch/'],
    'pamap2': ['/mnt/sting/hjyoon/projects/cross/PAMAP2/augcon/target_domain_chest/'],
    'dsa': ['/mnt/sting/hjyoon/projects/cross/DSA/augcon/target_domain_LA/']
}

TLR = {
    'ichar': 0.001,
    'hhar': 0.01,
    'pamap2': 0.01,
    'dsa': 0.001,
}

SHOT = 2

NUM_CLS = {'ichar': 9,
           'hhar': 6,
           'pamap2': 12,
           'dsa': 19}

PRETRAIN_MODEL_PATH = '/mnt/sting/hjyoon/projects/aaa/models/imwut/main_eval'
CONFIG_PATH = '/mnt/sting/hjyoon/projects/aaa/configs/imwut/main_hps_finetune_fix'
MODEL_PATH = '/mnt/sting/hjyoon/projects/aaa/models/imwut/main_hps_finetune_fix'


def gen_finetune_config():
    parameters = []
    for lr in PRETRAIN_HPS['lr']:
        for epoch in PRETRAIN_HPS['epoch']:
            parameters.append((lr, epoch))

    gpu = 0
    for dataset in DATASETS:
        data_paths = DATA_PATH[dataset]
        tlr = TLR[dataset]
        for data_path in data_paths:
            domain = data_path.split('/')[-2]
            port = 8567 + gpu
            for param in parameters:
                lr, epoch = param
                param_str = f'lr{lr}_ep{epoch}'

                num_cls = NUM_CLS[dataset]
                pretrain_ckpt_path = f'{PRETRAIN_MODEL_PATH}/{dataset}/{PRETEXT}/pretrain/{domain}'

                finetune_config_path = f'{CONFIG_PATH}/{dataset}/{PRETEXT}/finetune/{param_str}/gpu{gpu}_{domain}.yaml'
                print(f'Generating {finetune_config_path}')

                finetune_path = f'{data_path}finetune/{SHOT}shot/target'
                finetune_ckpt_path = f'{MODEL_PATH}/{dataset}/{PRETEXT}/finetune/{param_str}/{domain}'
                pretrained_path = f'{pretrain_ckpt_path}/checkpoint_{epoch}.pth.tar'
                finetune_config = get_config('finetune', [gpu], port, dataset,
                                             finetune_path, num_cls, 'crossentropy',
                                             lr, tlr, SHOT,
                                             finetune_ckpt_path,
                                             pretrained_path)

                os.makedirs(os.path.dirname(finetune_config_path), exist_ok=True)
                with open(finetune_config_path, 'w') as f:
                    f.write(finetune_config)
            gpu += 1
            if gpu == 8: gpu = 0


def get_config(mode, gpu, port, dataset_name, data_path, num_cls,
               criterion, lr, tlr, batch_size, ckpt_path, pretrained):
    config = f'''mode: {mode}
seed: 0
gpu: {gpu}
num_workers: {8 if mode == 'pretrain' else 1}
dist_url: tcp://localhost:{port}
episodes: 1

dataset_name: {dataset_name}
train_dataset_path: {data_path}/train.pkl
test_dataset_path: {data_path}/val.pkl
val_dataset_path: {data_path}/val.pkl
input_channels: 3
num_cls: {num_cls}

optimizer: adam
criterion: {criterion}
start_epoch: 0
epochs: 20

batch_size: {batch_size}
lr: {lr}
wd: 0

resume: ''
pretrained: {pretrained}
ckpt_dir: {ckpt_path}
log_freq: 100
save_freq: {1000 if mode == 'pretrain' else 10}

task_per_domain: true
num_task: 8
multi_cond_num_task: 4
task_size: 128
task_lr: {tlr}
reg_lambda: 0
log_meta_train: false

pretext: {PRETEXT}

out_dim: 50
T: 0.1
z_dim: 96
mlp: {'true' if mode == 'pretrain' else 'false'}

neg_per_domain: false
freeze: true
domain_adaptation: true
out_cls_neg_sampling: false
task_steps: 10
no_vars: true
'''
    return config


if __name__ == '__main__':
    gen_finetune_config()

    # rm -rf /mnt/sting/hjyoon/projects/aaa/configs/imwut/main_hps_finetune_fix/*/simclr/finetune
    # rm -rf /mnt/sting/hjyoon/projects/aaa/configs/imwut/main_hps_finetune_fix/*/metasimclr/finetune