import os
from glob import glob

PRETEXT = "simclr"
PRETRAIN_CRITERION = "crossentropy"
PRETRAIN_HPS = {
    "lr": [0.001, 0.0005, 0.0001],
    "wd": [0.0],
    "bs": [1024, 2048, 4096],
}

DATASETS = ["wesad", "ninaprodb5"]
DATA_PATH = {
    "wesad": ["/mnt/sting/hjyoon/projects/cross/WESAD/augcon/target_domain_S2/"],
    "ninaprodb5": [
        "/mnt/sting/hjyoon/projects/cross/NinaproDB5/augcon/target_domain_s1/"
    ],
}

NUM_CLS = {
    "wesad": 3,
    "ninaprodb5": 40,
}

INPUT_CHANNELS = {
    "wesad": 1,
    "ninaprodb5": 16,
}

CONFIG_PATH = "/mnt/sting/hjyoon/projects/aaa/configs/infocom/main_hps"
MODEL_PATH = "/mnt/sting/hjyoon/projects/aaa/models/infocom/main_hps"


def gen_pretrain_config():
    parameters = []
    for lr in PRETRAIN_HPS["lr"]:
        for wd in PRETRAIN_HPS["wd"]:
            for bs in PRETRAIN_HPS["bs"]:
                parameters.append((lr, wd, bs))

    gpu = 0
    for dataset in DATASETS:
        data_paths = DATA_PATH[dataset]
        for data_path in data_paths:
            domain = data_path.split("/")[-2]
            port = 7567 + gpu
            for param in parameters:
                param_str = f"lr{param[0]}_wd{param[1]}_bs{param[2]}"
                pretrain_config_path = f"{CONFIG_PATH}/{dataset}/{PRETEXT}/pretrain/{param_str}/gpu{gpu}_{domain}.yaml"
                print(f"Generating {pretrain_config_path}")

                pretrain_path = f"{data_path}pretrain"
                num_cls = NUM_CLS[dataset]
                input_channels = INPUT_CHANNELS[dataset]
                epochs = 50
                lr, wd, bs = param
                pretrain_ckpt_path = (
                    f"{MODEL_PATH}/{dataset}/{PRETEXT}/pretrain/{param_str}/{domain}"
                )
                pretrain_config = get_config(
                    "pretrain",
                    [gpu],
                    port,
                    dataset,
                    pretrain_path,
                    input_channels,
                    num_cls,
                    PRETRAIN_CRITERION,
                    epochs,
                    bs,
                    lr,
                    wd,
                    pretrain_ckpt_path,
                    None,
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
    ckpt_path,
    pretrained,
):
    config = f"""mode: {mode}
seed: 0
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
save_freq: 10

pretext: {PRETEXT if mode == 'pretrain' else 'meta' + PRETEXT}

out_dim: 50
T: 0.1
z_dim: 96

neg_per_domain: false
mlp: {'true' if mode == 'pretrain' else 'false'}
freeze: true
domain_adaptation: false
task_steps: -1
task_lr: -1
reg_lambda: -1
no_vars: true
"""
    return config


if __name__ == "__main__":
    gen_pretrain_config()
