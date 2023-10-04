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
    
    configs = defaultdict(list)
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    for domain in domains:
        w_configs = glob(os.path.join(config_dir, f'pt_w_{domain}/SimSiam_DomainNet_ft_{domain}.yaml'))
        wo_configs = glob(os.path.join(config_dir, f'pt_wo_{domain}/SimSiam_DomainNet_ft_{domain}.yaml'))
        for config in w_configs+wo_configs:
            with open(config, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'gpu' in line:
                        gpu = int(line.split(':')[-1].strip().split('[')[-1].split(']')[0])
                        configs[gpu].append(config)
                        break
    
    processes = []
    for gpu, config in configs.items():
        processes.append(mp.Process(target=execute_script, args=(gpu, python_script, config)))
    
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
