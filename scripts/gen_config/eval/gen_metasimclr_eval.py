import os
from glob import glob

PRETEXT = 'metasimclr'
PRETRAIN_CRITERION = 'crossentropy'
PRETRAIN_HPS = {
    'ichar': {'lr': 0.001, 'wd': 0.0, 'tlr': 0.001},
    'hhar': {'lr': 0.001, 'wd': 0.0001, 'tlr': 0.01},
    'pamap2': {'lr': 0.001, 'wd': 0.0, 'tlr': 0.01},
    'dsa': {'lr': 0.005, 'wd': 0.0, 'tlr': 0.001},
}

DATASETS = ['ichar', 'hhar', 'pamap2', 'dsa']
DATA_PATH = {
    'ichar': '/mnt/sting/hjyoon/projects/cross/ICHAR/augcon/',
    'hhar': '/mnt/sting/hjyoon/projects/cross/HHAR/augcon/',
    'pamap2': '/mnt/sting/hjyoon/projects/cross/PAMAP2/augcon/',
    'dsa': '/mnt/sting/hjyoon/projects/cross/DSA/augcon/'
}

NUM_CLS = {'ichar': 9,
           'hhar': 6,
           'pamap2': 12,
           'dsa': 19}

CONFIG_PATH = '/mnt/sting/hjyoon/projects/aaa/configs/imwut/main_eval'
MODEL_PATH = '/mnt/sting/hjyoon/projects/aaa/models/imwut/main_eval'


def gen_pretrain_config():
    for dataset in DATASETS:
        data_path = DATA_PATH[dataset]
        param = PRETRAIN_HPS[dataset]
        domains = glob(os.path.join(data_path, '*'))
        domains = [os.path.basename(domain) for domain in domains]
        gpu = 0
        for domain in domains:
            port = 8367 + gpu
            pretrain_config_path = f'{CONFIG_PATH}/{dataset}/{PRETEXT}/pretrain/gpu{gpu}_{domain}.yaml'
            print(f'Generating {pretrain_config_path}')

            pretrain_path = f'{data_path}{domain}/pretrain'
            num_cls = NUM_CLS[dataset]
            epochs = 5000
            lr, wd, tlr = param['lr'], param['wd'], param['tlr']
            pretrain_ckpt_path = f'{MODEL_PATH}/{dataset}/{PRETEXT}/pretrain/{domain}'
            pretrain_config = get_config('pretrain', [gpu], port, dataset,
                                         pretrain_path, num_cls, PRETRAIN_CRITERION,
                                         epochs, -1, lr, wd, tlr,
                                         pretrain_ckpt_path, None, True, 0)

            os.makedirs(os.path.dirname(pretrain_config_path), exist_ok=True)
            with open(pretrain_config_path, 'w') as f:
                f.write(pretrain_config)

            for seed in [0,1,2,3,4]:
                for shot in [1, 2, 5, 10, 20]:
                    for freeze in [True]:
                        setting = 'linear' if freeze else 'endtoend'
                        finetune_config_path = f'{CONFIG_PATH}/{dataset}/{PRETEXT}/finetune/{shot}shot/{setting}/seed{seed}/gpu{gpu}_{domain}.yaml'
                        print(f'Generating {finetune_config_path}')

                        finetune_path = f'{data_path}{domain}/finetune/{shot}shot/target'
                        finetune_ckpt_path = f'{MODEL_PATH}/{dataset}/{PRETEXT}/finetune/{shot}shot/{setting}/seed{seed}/{domain}'
                        pretrained_path = f'{pretrain_ckpt_path}/checkpoint_4999.pth.tar'
                        finetune_config = get_config('finetune', [gpu], port, dataset,
                                                    finetune_path, num_cls, 'crossentropy',
                                                    50, 4, 0.001, 0.0, tlr,
                                                    finetune_ckpt_path,
                                                    pretrained_path, freeze, seed)

                        os.makedirs(os.path.dirname(finetune_config_path), exist_ok=True)
                        with open(finetune_config_path, 'w') as f:
                            f.write(finetune_config)
            gpu += 1
            if gpu == 8: gpu = 0


def get_config(mode, gpu, port, dataset_name, data_path, num_cls,
               criterion, epochs, bs, lr, wd, tlr, ckpt_path, pretrained, freeze, seed):
    config = f'''mode: {mode}
seed: {seed}
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
epochs: {epochs}

batch_size: {bs}
lr: {lr}
wd: {wd}

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
freeze: {'true' if freeze else 'false'}
domain_adaptation: true
out_cls_neg_sampling: false
task_steps: 10
no_vars: true
'''
    return config


if __name__ == '__main__':
    gen_pretrain_config()