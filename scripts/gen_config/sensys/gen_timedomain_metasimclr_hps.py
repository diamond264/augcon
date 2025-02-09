import os
from glob import glob

PRETEXT = "metasimclr"
PRETRAIN_CRITERION = "crossentropy"
PRETRAIN_HPS = {
    'lr': [0.005, 0.01, 0.001],
    'wd': [0.0, 0.0001],
    'tlr': [0.005, 0.001, 0.01],
}

DATASETS = ["opportunity", "ninaprodb5"]
DATA_PATH = {
    "opportunity": "/mnt/sting/hjyoon/projects/cross/Opportunity/augcon_timedomain/target_domain_1",
    "ninaprodb5": "/mnt/sting/hjyoon/projects/cross/NinaproDB5/augcon_timedomain/target_domain_1",
}

NUM_CLS = {
    "opportunity": 4,
    "ninaprodb5": 12,
}

INPUT_CHANNELS = {
    "opportunity": 3,
    "ninaprodb5": 16,
}

CONFIG_PATH = "/mnt/sting/hjyoon/projects/aaa/configs/sensys/hps/timedomain"
MODEL_PATH = "/mnt/sting/hjyoon/projects/aaa/models/sensys/hps/timedomain"


def gen_pretrain_config():
    parameters = []
    for lr in PRETRAIN_HPS['lr']:
        for wd in PRETRAIN_HPS['wd']:
            for tlr in PRETRAIN_HPS['tlr']:
                parameters.append((lr, wd, tlr))
                
    for dataset in DATASETS:
        data_path = DATA_PATH[dataset]
        gpu = 0
        port = 8367 + gpu
        for param in parameters:
            param_str = f'lr{param[0]}_wd{param[1]}_tlr{param[2]}'
            pretrain_config_path = (
                f"{CONFIG_PATH}/{dataset}/{PRETEXT}/pretrain/{param_str}/gpu{gpu}_hps.yaml"
            )
            print(f"Generating {pretrain_config_path}")

            pretrain_path = f"{data_path}/pretrain"
            num_cls = NUM_CLS[dataset]
            epochs = 1000
            lr, wd, tlr = param
            pretrain_ckpt_path = f"{MODEL_PATH}/{dataset}/{PRETEXT}/pretrain/{param_str}/"
            pretrain_config = get_config(
                "pretrain",
                [gpu],
                port,
                dataset,
                pretrain_path,
                INPUT_CHANNELS[dataset],
                num_cls,
                PRETRAIN_CRITERION,
                epochs,
                -1,
                lr,
                wd,
                tlr,
                pretrain_ckpt_path,
                None,
                None,
                0.1,
                0,
            )

            os.makedirs(os.path.dirname(pretrain_config_path), exist_ok=True)
            with open(pretrain_config_path, "w", encoding="utf-8") as f:
                f.write(pretrain_config)

            gpu += 1
            if gpu == 8:
                gpu = 0


def get_config(
    mode,
    gpu,
    port,
    dataset_name,
    data_path,
    input_channels,
    num_cls,
    criterion,
    epochs,
    bs,
    lr,
    wd,
    tlr,
    ckpt_path,
    pretrained,
    pretrained_membank,
    mlr,
    seed,
):
    config = f"""mode: {mode}
seed: {seed}
gpu: {gpu}
num_workers: {8 if mode == 'pretrain' else 1}
dist_url: tcp://localhost:{port}
episodes: 1

dataset_name: {dataset_name}
train_dataset_path: {data_path}/train.pkl
test_dataset_path: {data_path}/val.pkl
val_dataset_path: {data_path}/val.pkl
input_channels: {input_channels}
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

pretrained_membank: {pretrained_membank}
bank_size: 1024
membank_lr: {mlr}
membank_optimizer: adam
membank_wd: 0.0
membank_m: 0.9
task_membank_lr: 1

pretext: {PRETEXT}

out_dim: 50
T: 0.1
z_dim: 96
mlp: {'true' if mode == 'pretrain' else 'false'}

neg_per_domain: false
freeze: true
domain_adaptation: true
adapt_w_neg: {'true' if mode == 'pretrain' else 'true'}
out_cls_neg_sampling: false
task_steps: 10
no_vars: true
visualization: false
"""
    return config


if __name__ == "__main__":
    gen_pretrain_config()
