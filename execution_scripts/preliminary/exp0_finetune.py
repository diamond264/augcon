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

def run(config_dir, python_script):
    if not os.path.exists(config_dir) or not os.path.exists(python_script):
        raise FileNotFoundError(f'{config_dir} or {python_script} does not exist')
    
    domains = glob(os.path.join(config_dir, 'finetune*/without_*'))
    domains = list(set([domain.split('/')[-1].split('without_')[1] for domain in domains]))
    
    domain_idx = {}
    gpu_idx = 0
    for domain in domains:
        domain_idx[domain] = gpu_idx
        gpu_idx += 1
        if gpu_idx == 8: gpu_idx = 0
    
    configs = glob(os.path.join(config_dir, 'finetune*/without_*/*/*.yaml'))
    config_per_gpu = defaultdict(list)
    for config in configs:
        domain = '_'.join(config.split('/')[-3].split('_')[1:])
        gpu = domain_idx[domain]
        config_per_gpu[gpu].append(config)
    
    processes = []
    for gpu, config_list in config_per_gpu.items():
        processes.append(mp.Process(target=execute_script, args=(gpu, python_script, config_list)))
    
    for process in processes:
        process.start()
    for process in processes:
        process.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'perform finetuning for exp0')
    parser.add_argument('--config_dir', type=str, required=True, help='config directory')
    python_script = '/home/hjyoon/projects/augcon/experiment.py'
    parser.add_argument('--python_script', type=str, default=python_script, help='python script to run')
    args = parser.parse_args()
    
    run(args.config_dir, args.python_script)
