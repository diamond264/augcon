#!/usr/bin/env python

import os
import argparse
import subprocess
import multiprocessing as mp

from glob import glob
from collections import defaultdict

def execute_script(gpu, script, config_list):
    for config in config_list:
        print(f'gpu{gpu} started {config}')
        subprocess.run(['python', script, '--config', config])
        print(f'gpu{gpu} finished {config}')

def run(config_dir, python_script, workers, worker_idx):
    if not os.path.exists(config_dir) or not os.path.exists(python_script):
        raise FileNotFoundError(f'{config_dir} or {python_script} does not exist')
        
    configs = glob(os.path.join(config_dir, 'pretrain/without_*/*.yaml'))
    config_per_gpu = defaultdict(list)
    for config in configs:
        domain = '_'.join(config.split('/')[-2].split('_')[1:])
        if 'perdomain' in config:
            if domain in ['target_domain_PH0007-jskim', 'target_domain_PH0012-thanh',
                          'target_domain_PH0014-wjlee', 'target_domain_PH0034-ykha',
                          'target_domain_PH0038-iygoo']:
                gpu = '0,1'
            else:
                gpu = '2,3'
        if 'random' in config:
            if domain in ['target_domain_PH0007-jskim', 'target_domain_PH0012-thanh',
                          'target_domain_PH0014-wjlee', 'target_domain_PH0034-ykha',
                          'target_domain_PH0038-iygoo']:
                gpu = '4,5'
            else:
                gpu = '6,7'
        config_per_gpu[gpu].append(config)
    
    processes = []
    # print(config_per_gpu)
    for gpu, config_list in config_per_gpu.items():
        configs = config_list[worker_idx::workers]
        processes.append(mp.Process(target=execute_script, args=(gpu, python_script, configs)))
    
    for process in processes:
        process.start()
    for process in processes:
        process.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'perform pretraining for exp0')
    parser.add_argument('--config_dir', type=str, required=True, help='config directory')
    python_script = '/home/hjyoon/projects/augcon/experiment.py'
    parser.add_argument('--python_script', type=str, default=python_script, help='python script to run')
    parser.add_argument('--workers', type=int, default=1, help='number of workers')
    parser.add_argument('--worker_idx', type=int, default=0, help='worker index')
    args = parser.parse_args()
    
    run(args.config_dir, args.python_script, args.workers, args.worker_idx)
