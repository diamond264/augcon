import os
from glob import glob

data_path = '/mnt/sting/hjyoon/projects/cross/ICHAR/augcon'
config_path = '/home/hjyoon/projects/augcon/config/exp0/pretrain'

def run():
    domains = glob(os.path.join(data_path, '*'))
    domains = [os.path.basename(domain) for domain in domains]
    print(domains)
    pretexts = ['cpc', 'metacpc']
    
    for architecture in ['8421']:
        for pretext in pretexts:
            for domain in domains:
                for perdomain in [True, False]:
                    if perdomain:
                        default_config = f'''### Default config
mode: pretrain
seed: 0
gpu: [0,1,2,3]
num_workers: 8
dist_url: tcp://localhost:10001
'''
                    else:
                        default_config = f'''### Default config
mode: pretrain
seed: 0
gpu: [4,5,6,7]
num_workers: 8
dist_url: tcp://localhost:10002
'''
                    dataset_config = f'''### Dataset config
dataset_name: ichar
train_dataset_path: /mnt/sting/hjyoon/projects/cross/ICHAR/augcon/{domain}/pretrain/train.pkl
test_dataset_path: /mnt/sting/hjyoon/projects/cross/ICHAR/augcon/{domain}/pretrain/test.pkl
val_dataset_path: /mnt/sting/hjyoon/projects/cross/ICHAR/augcon/{domain}/pretrain/val.pkl
input_channels: 3
num_cls: 9
'''
                    epochs = 100 if pretext == 'cpc' else 1000
                    training_config = f'''### Training config
optimizer: adam
criterion: crossentropy
start_epoch: 0
epochs: {epochs}
batch_size: 2048
lr: 0.0005
wd: 0.0
'''
                    ckpt_name = f'{pretext}_perdomain_{architecture}' if perdomain else f'{pretext}_random_{architecture}'
                    save_freq = 10 if pretext == 'cpc' else 100
                    log_config = f'''### Logs and checkpoints
resume: ''
ckpt_dir: /mnt/sting/hjyoon/projects/augcontrast/models/exp0/pretrain/without_{domain}/{ckpt_name}
log_freq: 20
save_freq: {save_freq}
'''
                    if architecture == '8421':
                        kernel_sizes = '[8, 4, 2, 1]'
                        strides = '[4, 2, 1, 1]'
                    elif architecture == '4111':
                        kernel_sizes = '[4, 1, 1, 1]'
                        strides = '[2, 1, 1, 1]'
                    neg_per_domain = 'true' if perdomain else 'false'
                    if pretext == 'cpc':
                        learning_config = f'neg_per_domain: {neg_per_domain}'
                    elif pretext == 'metacpc':
                        num_task = 10 if perdomain else 'null'
                        multi_cond_num_task = 5 if perdomain else 15
                        learning_config = f'''### Meta-learning
task_per_domain: {neg_per_domain}
num_task: {num_task}
multi_cond_num_task: {multi_cond_num_task}
task_size: 90
task_steps: 10
task_lr: 0.001
reg_lambda: 0
log_meta_train: false'''
                    model_config = f'''### Model config
pretext: {pretext}
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
                    file_path = os.path.join(config_path, f'without_{domain}', f'{pretext}_{perdom_type}_{architecture}.yaml')
                    # generate directory if not exist
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    # generate config file
                    with open(file_path, 'w') as f:
                        f.write(config)

if __name__ == '__main__':
    run()