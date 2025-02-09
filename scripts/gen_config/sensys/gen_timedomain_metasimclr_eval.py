import os
from glob import glob

PRETEXT = "metasimclr"
PRETRAIN_CRITERION = "crossentropy"
PRETRAIN_HPS = {
    "opportunity": {"lr": 0.001, "wd": 0.0, "tlr": 0.001}, #####
    "ninaprodb5": {"lr": 0.005, "wd": 0.0, "tlr": 0.001},
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
            epochs = 5000
            lr, wd, tlr = param["lr"], param["wd"], param["tlr"]
            pretrain_ckpt_path = f"{MODEL_PATH}/{dataset}/{PRETEXT}/pretrain/{domain}"
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

            for seed in [0, 1, 2, 3, 4]:
                for shot in [1, 2, 5, 10]:
                    for freeze in [True]:
                        setting = "linear" if freeze else "endtoend"
                        finetune_config_path = f"{CONFIG_PATH}/{dataset}/{PRETEXT}/finetune/{shot}shot/{setting}/seed{seed}/gpu{gpu}_{domain}.yaml"
                        print(f"Generating {finetune_config_path}")

                        finetune_path = (
                            f"{data_path}{domain}/finetune/{shot}shot/target"
                        )
                        finetune_ckpt_path = f"{MODEL_PATH}/{dataset}/{PRETEXT}/finetune/{shot}shot/{setting}/seed{seed}/{domain}"
                        pretrained_path = (
                            f"{pretrain_ckpt_path}/checkpoint_4999.pth.tar"
                        )
                        pretrained_membank_path = (
                            f"{pretrain_ckpt_path}/membank_4999.pth.tar"
                        )
                        ft_lr = 0.005 if freeze else 0.001
                        bs = 4 if shot != 1 else 1
                        finetune_config = get_config(
                            "finetune",
                            [gpu],
                            port,
                            dataset,
                            finetune_path,
                            INPUT_CHANNELS[dataset],
                            num_cls,
                            "crossentropy",
                            20,
                            bs,
                            ft_lr,
                            0.0,
                            tlr,
                            finetune_ckpt_path,
                            pretrained_path,
                            pretrained_membank_path,
                            freeze,
                            seed,
                        )

                        os.makedirs(
                            os.path.dirname(finetune_config_path), exist_ok=True
                        )
                        with open(finetune_config_path, "w", encoding="utf-8") as f:
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
