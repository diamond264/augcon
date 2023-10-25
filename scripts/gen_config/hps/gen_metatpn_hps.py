import os
from glob import glob

PRETEXT = 'metatpn'
PRETRAIN_CRITERION = 'crossentropy'
PRETRAIN_HPS = {
    'lr': [0.0005, 0.001, 0.005],
    'wd': [0.0, 0.0001],
    'tlr': [0.001, 0.005, 0.01],
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

NUM_CLS = {'ichar': 9,
           'hhar': 6,
           'pamap2': 12,
           'dsa': 19}

CONFIG_PATH = '/mnt/sting/hjyoon/projects/aaa/configs/imwut/main_hps'
MODEL_PATH = '/mnt/sting/hjyoon/projects/aaa/models/imwut/main_hps'

def gen_pretrain_config():
    parameters = []
    for lr in PRETRAIN_HPS['lr']:
        for wd in PRETRAIN_HPS['wd']:
            for tlr in PRETRAIN_HPS['tlr']:
                parameters.append((lr, wd, tlr))
    
    gpu = 0
    for dataset in DATASETS:
        data_paths = DATA_PATH[dataset]
        for data_path in data_paths:
            domain = data_path.split('/')[-2]
            port = 8678 + gpu
            for param in parameters:
                param_str = f'lr{param[0]}_wd{param[1]}_tlr{param[2]}'
                pretrain_config_path = f'{CONFIG_PATH}/{dataset}/{PRETEXT}/pretrain/{param_str}/gpu{gpu}_{domain}.yaml'
                print(f'Generating {pretrain_config_path}')

                pretrain_path = f'{data_path}pretrain'
                num_cls = NUM_CLS[dataset]
                epochs = 3000
                lr, wd, tlr = param
                pretrain_ckpt_path = f'{MODEL_PATH}/{dataset}/{PRETEXT}/pretrain/{param_str}/{domain}'
                pretrain_config = get_config('pretrain', [gpu], port, dataset,
                                             pretrain_path, num_cls, PRETRAIN_CRITERION,
                                             epochs, -1, lr, wd, tlr,
                                             pretrain_ckpt_path, None)
                
                os.makedirs(os.path.dirname(pretrain_config_path), exist_ok=True)
                with open(pretrain_config_path, 'w') as f:
                    f.write(pretrain_config)
                
                finetune_config_path = f'{CONFIG_PATH}/{dataset}/{PRETEXT}/finetune/{param_str}/gpu{gpu}_{domain}.yaml'
                print(f'Generating {finetune_config_path}')
                
                finetune_path = f'{MODEL_PATH}/{data_path}finetune/10shot/target'
                finetune_ckpt_path = f'{MODEL_PATH}/{dataset}/{PRETEXT}/finetune/{param_str}/{domain}'
                pretrained_path = f'{pretrain_ckpt_path}/checkpoint_2999.pth.tar'
                finetune_config = get_config('finetune', [gpu], port, dataset,
                                             finetune_path, num_cls, 'crossentropy',
                                             50, 4, 0.001, 0.0, tlr,
                                             finetune_ckpt_path,
                                             pretrained_path)
                
                os.makedirs(os.path.dirname(finetune_config_path), exist_ok=True)
                with open(finetune_config_path, 'w') as f:
                    f.write(finetune_config)
            gpu += 1
            if gpu == 8: gpu = 0


def get_config(mode, gpu, port, dataset_name, data_path, num_cls,
               criterion, epochs, bs, lr, wd, tlr, ckpt_path, pretrained):
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
out_dim: 2
z_dim: 96
neg_per_domain: false

mlp: false
freeze: true
domain_adaptation: true
task_steps: 10
no_vars: true
'''
    return config

if __name__ == '__main__':
    gen_pretrain_config()