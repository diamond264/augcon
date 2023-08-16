import os
import argparse
from glob import glob
from collections import defaultdict

def run(config_dir):
    ### OPTIONS
    configs = glob(os.path.join(config_dir, '*.yaml.log'))
    
    in_domain_res = []
    out_domain_res = []
    random_res = []
    
    for config in configs:
        source = config.split('/')[-1].split('finetune')[1].split('_')[0]
        if source == '': source = 'random'
        target = config.split('/')[-1].split('finetune')[1].split('_')[-1].split('.')[0]
        
        with open(config, 'r') as f:
            content = f.read()
            if content == "":
                print(f'no file {config}')
                continue
        
        if 'Validation Loss: ' not in content:
            print(f'no validation loss in {config}')
            continue
        
        acc = content.split('Validation Loss: ')[-1].split('Acc(1): ')[1].split(', Acc(5): ')[0]
        acc = float(acc)
        if source == 'random': random_res.append(acc)
        elif source == '1' and target == '3': out_domain_res.append(acc)
        elif source == '2' and target == '2': out_domain_res.append(acc)
        elif source == '3' and target == '1': out_domain_res.append(acc)
        else: in_domain_res.append(acc)
    
    print(f'random: {sum(random_res)/len(random_res):.4f}')
    print(f'in domain: {sum(in_domain_res)/len(in_domain_res):.4f}')
    print(f'out domain: {sum(out_domain_res)/len(out_domain_res):.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'analyze results for exp1')
    parser.add_argument('--config_dir', type=str, required=True, help='config directory')
    args = parser.parse_args()
    
    run(args.config_dir)