import os

shots = ['1', '2', '5', '10']
users = ['a', 'b', 'c', 'd', 'f']
models = ['nexus4', 's3', 's3mini', 'lgwatch']

for user in users:
    for model in models:
        config = '''seed: null
gpu: null
data:
  train_dataset_path: /mnt/sting/hjyoon/projects/cross/HHAR/augcon/target_user_'''+user+'''_model_'''+model+'''/pretrain/train.pkl
  test_dataset_path: /mnt/sting/hjyoon/projects/cross/HHAR/augcon/target_user_'''+user+'''_model_'''+model+'''/pretrain/test.pkl
  val_dataset_path: /mnt/sting/hjyoon/projects/cross/HHAR/augcon/target_user_'''+user+'''_model_'''+model+'''/pretrain/val.pkl
  name: hhar
multiprocessing:
  multiprocessing_distributed: true
  workers: 32
  world_size: 1
  rank: 0
  dist_url: tcp://localhost:10001
  dist_backend: nccl
train:
  optimizer: adam
  criterion: crossentropy
  resume: ""
  epochs: 100
  start_epoch: 0
  batch_size: 256
  lr: 0.0001
  fix_pred_lr: false
  weight_decay: 0.0001
  print_freq: 100
  save_dir: /mnt/sting/hjyoon/projects/augcontrast/models/pretrained/230418_target_user_'''+user+'''_model_'''+model+'''
  cos: false
  schedule: []
model:
  type: cpc
  input_channels: 3
  z_dim: 256
  num_blocks: 5
  num_filters: 256
  pred_steps: 12
  n_negatives: 15
  offset: 16
'''
        filename = 'target_user_'+user+'_model_'+model
        if not os.path.exists('/home/hjyoon/projects/augcon/config/pretrain/preliminary'):
            os.makedirs('/home/hjyoon/projects/augcon/config/pretrain/preliminary')
            
        with open('/home/hjyoon/projects/augcon/config/pretrain/preliminary/'+filename+'.yaml', 'w') as f:
            f.write(config)