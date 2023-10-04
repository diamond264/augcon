import os

shots = ['1', '2', '5', '10']
users = ['a', 'b', 'c', 'd', 'f']
models = ['nexus4', 's3', 's3mini', 'lgwatch']
domains = ['target']

for domain in domains:
    for shot in shots:
        for user in users:
            for model in models:
                for user2 in users:
                    for model2 in models:
                        if model2==model and user2!=user: continue
                        if model2!=model and user2==user: continue
                        config = '''seed: null
gpu: 0
evaluate: false
data:
    name: hhar
    num_cls: 6
    train_dataset_path: /mnt/sting/hjyoon/projects/cross/HHAR/augcon/target_user_'''+user2+'''_model_'''+model2+'''/finetune/'''+shot+'''shot/'''+domain+'''/train.pkl
    test_dataset_path: /mnt/sting/hjyoon/projects/cross/HHAR/augcon/target_user_'''+user2+'''_model_'''+model2+'''/finetune/'''+shot+'''shot/'''+domain+'''/test.pkl
    val_dataset_path: /mnt/sting/hjyoon/projects/cross/HHAR/augcon/target_user_'''+user2+'''_model_'''+model2+'''/finetune/'''+shot+'''shot/'''+domain+'''/val.pkl
multiprocessing:
    multiprocessing_distributed: false
    workers: 32
    world_size: 1
    rank: 0
    dist_url: tcp://localhost:10002
    dist_backend: nccl
train:
    pretrained: /mnt/sting/hjyoon/projects/augcontrast/models/pretrained/230418_target_user_'''+user+'''_model_'''+model+'''/checkpoint_0099.pth.tar
    optimizer: adam
    criterion: crossentropy
    resume: ''
    epochs: 50
    start_epoch: 0
    batch_size: 4
    lr: 0.0005
    fix_pred_lr: false
    weight_decay: 0
    print_freq: 2
    save_dir: /mnt/sting/hjyoon/projects/augcontrast/models/finetuned/230418_pt_except_user_'''+user+'''_model_'''+model+'''_finetune_user_'''+user2+'''_model_'''+model2+'''_shot_'''+shot+'''
    cos: false
    schedule: []
test:
    batch_size: 2
    print_freq: 100
model:
    type: cpc
    input_channels: 3
    z_dim: 256
    num_blocks: 5
    num_filters: 256
    seq_len: 30'''
                        filename = '230418_pt_except_shot_'+shot+'_user_'+user+'_model_'+model+'_finetune_user_'+user2+'_model_'+model2
                        with open('/home/hjyoon/projects/augcon/config/finetune/preliminary/'+filename+'.yaml', 'w') as f:
                            f.write(config)
