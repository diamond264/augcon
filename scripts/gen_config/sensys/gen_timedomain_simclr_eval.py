import os
from glob import glob

PRETEXT = "simclr"
PRETRAIN_CRITERION = "crossentropy"
PRETRAIN_HPS = {
    "opportunity": {"lr": 0.001, "wd": 0.0, "bs": 64},
    "ninaprodb5": {"lr": 0.001, "wd": 0.0, "bs": 64},
}

DATASETS = ["opportunity", "ninaprodb5"]
DATA_PATH = {
    "opportunity": "/mnt/sting/hjyoon/projects/cross/Opportunity/augcon_timedomain/",
    "ninaprodb5": "/mnt/sting/hjyoon/projects/cross/NinaproDB5/augcon_timedomain/",
}

NUM_CLS = {
    "opportunity": 4,
    "ninaprodb5": 12,
}

INPUT_CHANNELS = {
    "opportunity": 3,
    "ninaprodb5": 16,
}

CONFIG_PATH = "/mnt/sting/hjyoon/projects/aaa/configs/sensys/shepherd/timedomain"
MODEL_PATH = "/mnt/sting/hjyoon/projects/aaa/models/sensys/shepherd/timedomain"


def gen_pretrain_config():
    for dataset in DATASETS:
        data_path = DATA_PATH[dataset]
        param = PRETRAIN_HPS[dataset]
        domains = glob(os.path.join(data_path, "*"))
        domains = [os.path.basename(domain) for domain in domains]
        gpu = 0
        for domain in domains:
            port = 8367 + gpu
            pretrain_config_path = (
                f"{CONFIG_PATH}/{dataset}/{PRETEXT}/pretrain/gpu{gpu}_{domain}.yaml"
            )
            print(f"Generating {pretrain_config_path}")

            pretrain_path = f"{data_path}{domain}/pretrain"
            num_cls = NUM_CLS[dataset]
            input_channels = INPUT_CHANNELS[dataset]
            epochs = 100
            lr, wd, bs = param["lr"], param["wd"], param["bs"]
            pretrain_ckpt_path = f"{MODEL_PATH}/{dataset}/{PRETEXT}/pretrain/{domain}"
            pretrain_config = get_config(
                "pretrain",
                [gpu],
                port,
                dataset,
                pretrain_path,
                num_cls,
                input_channels,
                PRETRAIN_CRITERION,
                epochs,
                bs,
                lr,
                wd,
                pretrain_ckpt_path,
                None,
                True,
                0,
            )

            os.makedirs(os.path.dirname(pretrain_config_path), exist_ok=True)
            with open(pretrain_config_path, "w") as f:
                f.write(pretrain_config)

            for seed in [0, 1, 2, 3, 4]:
                for shot in [1, 2, 5, 10, 20]:
                    for freeze in [True, False]:
                        setting = "linear" if freeze else "endtoend"
                        finetune_config_path = f"{CONFIG_PATH}/{dataset}/{PRETEXT}/finetune/{shot}shot/{setting}/seed{seed}/gpu{gpu}_{domain}.yaml"
                        print(f"Generating {finetune_config_path}")

                        finetune_path = (
                            f"{data_path}{domain}/finetune/{shot}shot/target"
                        )
                        finetune_ckpt_path = f"{MODEL_PATH}/{dataset}/{PRETEXT}/finetune/{shot}shot/{setting}/seed{seed}/{domain}"
                        pretrained_path = (
                            f"{pretrain_ckpt_path}/checkpoint_0099.pth.tar"
                        )
                        ft_lr = 0.005 if freeze else 0.001
                        bs = 4 if shot != 1 else 1
                        finetune_config = get_config(
                            "finetune",
                            [gpu],
                            port,
                            dataset,
                            finetune_path,
                            num_cls,
                            input_channels,
                            "crossentropy",
                            20,
                            bs,
                            ft_lr,
                            0.0,
                            finetune_ckpt_path,
                            pretrained_path,
                            freeze,
                            seed,
                        )

                        os.makedirs(
                            os.path.dirname(finetune_config_path), exist_ok=True
                        )
                        with open(finetune_config_path, "w") as f:
                            f.write(finetune_config)
            gpu += 1
            if gpu == 8:
                gpu = 0


def get_config(
    mode,
    gpu,
    port,
    dataset_name,
    data_path,
    num_cls,
    input_channels,
    criterion,
    epochs,
    bs,
    lr,
    wd,
    ckpt_path,
    pretrained,
    freeze,
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
save_freq: 10

pretext: {PRETEXT if mode == 'pretrain' else 'meta' + PRETEXT}
out_dim: 50
T: 0.1
z_dim: 96
neg_per_domain: false

pretrained_membank: no_file
bank_size: 1024
membank_lr: 0
membank_optimizer: adam
membank_wd: 0.0
membank_m: 0.9
task_membank_lr: 1

mlp: {'true' if mode == 'pretrain' else 'false'}
freeze: {'true' if freeze else 'false'}
domain_adaptation: false
task_steps: -1
task_lr: -1
reg_lambda: -1
no_vars: true
"""
    return config


if __name__ == "__main__":
    gen_pretrain_config()
