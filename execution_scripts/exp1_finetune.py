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
    
    configs = glob(os.path.join(config_dir, '*.yaml'))
    processes = []
    for gpu, config in enumerate(configs):
        config_list = [config]
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
