import os

shots = ['1', '2', '5', '10']
users = ['a', 'b', 'c', 'd', 'e', 'f']
models = ['nexus4', 's3', 's3mini', 'lgwatch']

for shot in shots:
    for except_u in users:
        for u in users:
            config = '''seed: null
gpu: 0
evaluate: false
data:
    path: /mnt/sting/hjyoon/projects/cross/HHAR/ActivityRecognitionExp/hhar_minmax_scaling_all.csv
    name: hhar
    class_type: gt
    num_cls: 6
    shot_num: '''+shot+'''
    test_size: 1000
    val_size: 500
    domain_type: user
    '''
        
            if u == except_u:
                config += '''save_cache: false
    load_cache: false'''
            else:
                config += '''save_cache: false
    load_cache: false'''
            config += '''
    split_ratio: 0
    save_opposite: ''
    user: '''+u+'''
    cache_path: /mnt/sting/hjyoon/projects/cross/HHAR/augcon/processed_hhar_finetune_except_user'''+except_u.upper()+'''.pkl
    fixed_data_path: /mnt/sting/hjyoon/projects/cross/HHAR/augcon/kshot_'''+shot+'''_dataset_target_user'''+u+'''.pkl
multiprocessing:
    multiprocessing_distributed: false
    workers: 32
    world_size: 1
    rank: 0
    dist_url: tcp://localhost:10002
    dist_backend: nccl
train:
    pretrained: /mnt/sting/hjyoon/projects/augcontrast/models/pretrained/230414_except_user'''+except_u+'''/checkpoint_0079.pth.tar
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
    save_dir: /mnt/sting/hjyoon/projects/augcontrast/models/finetuned/pretrained_except_'''+except_u+'''_finetuned_'''+u+'''_shot_'''+shot+'''
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
            out_dir = '/mnt/sting/hjyoon/projects/augcontrast/models/finetuned/pretrained_except_'+except_u+'_finetuned_'+u+'_shot_'+shot
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            filename = 'pretrained_except_'+except_u+'_finetuned_'+u+'_shot_'+shot
            with open('/home/hjyoon/projects/augcon/config/finetune/preliminary/'+filename+'.yaml', 'w') as f:
                f.write(config)

    for except_m in models:
        for m in models:
            config = '''seed: null
gpu: 0
evaluate: false
data:
    path: /mnt/sting/hjyoon/projects/cross/HHAR/ActivityRecognitionExp/hhar_minmax_scaling_all.csv
    name: hhar
    class_type: gt
    num_cls: 6
    shot_num: '''+shot+'''
    test_size: 1000
    val_size: 500
    domain_type: model
    '''
        
            if m == except_m:
                config += '''save_cache: false
    load_cache: false'''
            else:
                config += '''save_cache: false
    load_cache: false'''
            config += '''
    split_ratio: 0
    save_opposite: ''
    model: '''+m+'''
    cache_path: /mnt/sting/hjyoon/projects/cross/HHAR/augcon/processed_hhar_finetune_except_model'''+except_m[0].upper()+except_m[1:]+'''.pkl
    fixed_data_path: /mnt/sting/hjyoon/projects/cross/HHAR/augcon/kshot_'''+shot+'''_dataset_target_model'''+m+'''.pkl
multiprocessing:
    multiprocessing_distributed: false
    workers: 32
    world_size: 1
    rank: 0
    dist_url: tcp://localhost:10002
    dist_backend: nccl
train:
    pretrained: /mnt/sting/hjyoon/projects/augcontrast/models/pretrained/230414_except_model'''+except_m+'''/checkpoint_0079.pth.tar
    optimizer: adam
    criterion: crossentropy
    resume: ''
    epochs: 50
    start_epoch: 0
    batch_size: 2
    lr: 0.0005
    fix_pred_lr: false
    weight_decay: 0
    print_freq: 2
    save_dir: /mnt/sting/hjyoon/projects/augcontrast/models/finetuned/pretrained_except_'''+except_m+'''_finetuned_'''+m+'''_shot_'''+shot+'''
    cos: false
    schedule: []
test:
    batch_size: 4
    print_freq: 100
model:
    type: cpc
    input_channels: 3
    z_dim: 256
    num_blocks: 5
    num_filters: 256
    seq_len: 30'''
            out_dir = '/mnt/sting/hjyoon/projects/augcontrast/models/finetuned/pretrained_except_'+except_m+'_finetuned_'+m+'_shot_'+shot
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            filename = 'pretrained_except_'+except_m+'_finetuned_'+m+'_shot_'+shot
            with open('/home/hjyoon/projects/augcon/config/finetune/preliminary/'+filename+'.yaml', 'w') as f:
                f.write(config)

# fixed_data_path: /mnt/sting/hjyoon/projects/cross/HHAR/augcon/kshot_dataset_target_user'''+u+'''.pkl