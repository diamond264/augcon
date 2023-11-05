import os
from glob import glob

PRETEXT = 'metacpc'
PRETRAIN_CRITERION = 'crossentropy'
PRETRAIN_HPS = {
    'ichar': {'lr': 0.0005, 'wd': 0.0, 'tlr': 0.001},
    'hhar': {'lr': 0.0005, 'wd': 0.0001, 'tlr': 0.001},
    'pamap2': {'lr': 0.001, 'wd': 0.0001, 'tlr': 0.001},
    'dsa': {'lr': 0.001, 'wd': 0.0001, 'tlr': 0.001},
}
PRETRAIN_HPS = {
    'ichar': {'lr': 0.001, 'wd': 0.0001, 'tlr': 0.005},
    'hhar': {'lr': 0.001, 'wd': 0.0001, 'tlr': 0.001},
    'pamap2': {'lr': 0.001, 'wd': 0.0001, 'tlr': 0.001},
    'dsa': {'lr': 0.0005, 'wd': 0.0001, 'tlr': 0.001},
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

EPOCHS = {
    'target_domain_WA0002-bkkim': 3999,
    'target_domain_PH0007-jskim': 3999,
    'target_domain_WA0003-hskim': 4999,
    'target_domain_PH0012-thanh': 2999,
    'target_domain_WA4697-jhryu': 1999,
    'target_domain_PH0014-wjlee': 2999,
    'target_domain_PH0034-ykha': 1999,
    'target_domain_PH0038-iygoo': 2999,
    'target_domain_PH0041-hmkim': 2999,
    'target_domain_PH0045-sjlee': 3999,

    'target_user_b_model_lgwatch': 3999,
    'target_user_d_model_lgwatch': 3999,
    'target_user_a_model_nexus4': 4999,
    'target_user_c_model_nexus4': 2999,
    'target_user_f_model_nexus4': 4999,
    'target_user_a_model_s3': 4999,
    'target_user_c_model_s3': 4999,
    'target_user_f_model_s3': 2999,
    'target_user_a_model_s3mini': 2999,
    'target_user_c_model_s3mini': 4999,
    'target_user_f_model_s3mini': 4999,
    'target_user_a_model_lgwatch': 4999,
    'target_user_c_model_lgwatch': 4999,
    'target_user_f_model_lgwatch': 3999,
    'target_user_b_model_nexus4': 4999,
    'target_user_d_model_nexus4': 4999,
    'target_user_b_model_s3': 4999,
    'target_user_d_model_s3': 4999,
    'target_user_b_model_s3mini': 3999,
    'target_user_d_model_s3mini': 4999,
}

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

            for seed in [0,1,2,3,4]:
                for shot in [1, 2, 5, 10, 20]:
                    for freeze in [True]:
                        setting = 'linear' if freeze else 'endtoend'
                        finetune_config_path = f'{CONFIG_PATH}/{dataset}/{PRETEXT}/finetune/{shot}shot/{setting}/seed{seed}/gpu{gpu}_{domain}.yaml'
                        print(f'Generating {finetune_config_path}')

                        finetune_path = f'{data_path}{domain}/finetune/{shot}shot/target'
                        finetune_ckpt_path = f'{MODEL_PATH}/{dataset}/{PRETEXT}/finetune/{shot}shot/{setting}/seed{seed}/{domain}'
                        epoch = 4999
                        if domain in EPOCHS.keys():
                            epoch = EPOCHS[domain]
                        pretrained_path = f'{pretrain_ckpt_path}/checkpoint_{epoch}.pth.tar'
                        ft_lr = 0.005 if freeze else 0.001
                        finetune_config = get_config('finetune', [gpu], port, dataset,
                                                        finetune_path, num_cls, 'crossentropy',
                                                        20, 4, ft_lr, 0.0, tlr,
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
enc_blocks: 4
kernel_sizes: [8, 4, 2, 1]
strides: [4, 2, 1, 1]
agg_blocks: 5
z_dim: 256
pooling: mean
pred_steps: 12
n_negatives: 15
offset: 4
neg_per_domain: false

mlp: false
freeze: {'true' if freeze else 'false'}
domain_adaptation: true
out_cls_neg_sampling: false
task_steps: 10
no_vars: true
'''
    return config


if __name__ == '__main__':
    gen_pretrain_config()