import os
from glob import glob

PRETEXT = "metatpn"
PRETRAIN_CRITERION = "crossentropy"
PRETRAIN_HPS = {
    "lr": [0.0005, 0.001, 0.005],
    "wd": [0.0],
    "tlr": [0.001, 0.005, 0.01],
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
            for tlr in PRETRAIN_HPS["tlr"]:
                parameters.append((lr, wd, tlr))

    gpu = 0
    for dataset in DATASETS:
        data_paths = DATA_PATH[dataset]
        for data_path in data_paths:
            domain = data_path.split("/")[-2]
            port = 4678 + gpu
            for param in parameters:
                param_str = f"lr{param[0]}_wd{param[1]}_tlr{param[2]}"
                pretrain_config_path = f"{CONFIG_PATH}/{dataset}/{PRETEXT}/pretrain/{param_str}/gpu{gpu}_{domain}.yaml"
                print(f"Generating {pretrain_config_path}")

                pretrain_path = f"{data_path}pretrain"
                num_cls = NUM_CLS[dataset]
                epochs = 1000
                lr, wd, tlr = param
                pretrain_ckpt_path = (
                    f"{MODEL_PATH}/{dataset}/{PRETEXT}/pretrain/{param_str}/{domain}"
                )
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
                )

                os.makedirs(os.path.dirname(pretrain_config_path), exist_ok=True)
                with open(pretrain_config_path, "w") as f:
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
"""
    return config


if __name__ == "__main__":
    gen_pretrain_config()
