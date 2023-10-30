import os
from glob import glob

PRETEXT = 'simclr'
PRETRAIN_CRITERION = 'crossentropy'
PRETRAIN_HPS = {
    'lr': [0.0001, 0.001, 0.01, 0.1],
}

LR = {
    'linear' : 0.001,
    'endtoend' : 0.001,
}

SHOT = 2

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
        parameters.append(lr)

    gpu = 0
    for dataset in DATASETS:
        data_paths = DATA_PATH[dataset]
        for data_path in data_paths:
            domain = data_path.split('/')[-2]
            port = 8567 + gpu
            # for param in parameters:
            for freeze in [True, False]:
                setting = 'linear' if freeze else 'endtoend'
                lr = LR[setting]
                param_str = f'lr{lr}'

                num_cls = NUM_CLS[dataset]
                pretrain_ckpt_path = f'{PRETRAIN_MODEL_PATH}/{dataset}/{PRETEXT}/pretrain/{domain}'

                finetune_config_path = f'{CONFIG_PATH}/{dataset}/{PRETEXT}/finetune/{setting}/{param_str}/gpu{gpu}_{domain}.yaml'
                print(f'Generating {finetune_config_path}')

                finetune_path = f'{data_path}finetune/{SHOT}shot/target'
                finetune_ckpt_path = f'{MODEL_PATH}/{dataset}/{PRETEXT}/finetune/{setting}/{param_str}/{domain}'
                pretrained_path = f'{pretrain_ckpt_path}/checkpoint_0099.pth.tar'
                finetune_config = get_config('finetune', [gpu], port, dataset,
                                             finetune_path, num_cls, 'crossentropy',
                                             lr, SHOT, finetune_ckpt_path,
                                             pretrained_path, freeze)

                os.makedirs(os.path.dirname(finetune_config_path), exist_ok=True)
                with open(finetune_config_path, 'w') as f:
                    f.write(finetune_config)
            gpu += 1
            if gpu == 8: gpu = 0


def get_config(mode, gpu, port, dataset_name, data_path, num_cls,
               criterion, lr, batch_size, ckpt_path, pretrained, freeze):
    config = f'''mode: finetune
seed: 0
gpu: {gpu}
num_workers: 1
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
save_freq: 10

pretext: {'meta' + PRETEXT}

out_dim: 50
T: 0.1
z_dim: 96

neg_per_domain: false
mlp: {'true' if mode == 'pretrain' else 'false'}
freeze: {freeze}
domain_adaptation: false
task_steps: -1
task_lr: -1
reg_lambda: -1
no_vars: true
'''
    return config


if __name__ == '__main__':
    gen_finetune_config()
    # rm -rf /mnt/sting/hjyoon/projects/aaa/configs/imwut/main_hps_finetune_fix/*/metaautoencoder/finetune/*