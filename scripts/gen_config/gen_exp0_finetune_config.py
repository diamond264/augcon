import os
from glob import glob

data_path = '/mnt/sting/hjyoon/projects/cross/ICHAR/augcon'
config_path = '/home/hjyoon/projects/augcon/config/exp0/finetune'

domain_idx = {
    'without_target_domain_PH007-jskim': 0,
    'without_target_domain_PH0012-thanh': 1,
    'without_target_domain_PH0014-wjlee': 2,
    'without_target_domain_PH0034-ykha': 3,
    'without_target_domain_PH0038-iygoo': 4,
    'without_target_domain_PH0041-hmkim': 5,
    'without_target_domain_PH0045-sjlee': 6,
    'without_target_domain_WA0002-bkkim': 7,
    'without_target_domain_WA0003-hskim': 0,
    'without_target_domain_WA4697-jhryu': 1,
}

def run():
    domains = glob(os.path.join(data_path, '*'))
    domains = [os.path.basename(domain) for domain in domains]
    print(domains)
    pretexts = ['cpc', 'metacpc']
    
    for seed in [0]:
        for architecture in ['8421']:
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
dataset_name: hhar
train_dataset_path: /mnt/sting/hjyoon/projects/cross/ICHAR/augcon/{tgt_domain}/finetune/10shot/target/train.pkl
test_dataset_path: /mnt/sting/hjyoon/projects/cross/ICHAR/augcon/{tgt_domain}/finetune/10shot/target/test.pkl
val_dataset_path: /mnt/sting/hjyoon/projects/cross/ICHAR/augcon/{tgt_domain}/finetune/10shot/target/val.pkl
input_channels: 3
num_cls: 9
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
                        ckpt_name = f'{pretext}_perdomain_{architecture}' if perdomain else f'{pretext}_random_{architecture}'
                        save_freq = 10
                        log_config = f'''### Logs and checkpoints
resume: ''
pretrained: 
ckpt_dir: /mnt/sting/hjyoon/projects/augcontrast/models/exp0/pretrain/{ckpt_name}
log_freq: 5
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