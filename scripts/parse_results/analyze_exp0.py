import os
import argparse
from glob import glob
from collections import defaultdict

def run(config_dir):
    ### OPTIONS
    configs = glob(os.path.join(config_dir, 'finetune*/without_*/*/*.yaml'))
    das = list(set([c.split('/')[-4].split('finetune_')[1] for c in configs]))
    domains = list(set([c.split('/')[-3].split('without_')[1] for c in configs]))
    methods = list(set([c.split('/')[-2] for c in configs]))
    
    res_dict = {}
    for da in das:
        for method in methods:
            res_dict[(method, da)] = {
                'in_domain': [],
                'out_domain': []
            }
            for src_dom in domains:
                for tgt_dom in domains:
                    finetune_dir = 'finetune_' + da
                    log_file = os.path.join(config_dir, 
                                            finetune_dir, 
                                            f'without_{src_dom}',
                                            f'{method}',
                                            f'tgt_{tgt_dom}_seed0.yaml.log')
                    log_content = ""
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        if log_content == "":
                            print(f'no file {log_file}')
                            continue
                    
                    if 'Validation Loss: ' not in log_content:
                        print(f'no validation loss in {log_file}')
                        continue
                    acc = log_content.split('Validation Loss: ')[-1].split('Acc(1): ')[1].split(', Acc(5): ')[0]
                    acc = float(acc)
                    
                    if src_dom == tgt_dom:
                        res_dict[(method, da)]['out_domain'].append(acc)
                    else:
                        res_dict[(method, da)]['in_domain'].append(acc)
    
    for dom in ['in_domain', 'out_domain']:
        print(f'==================== {dom} ====================')
        for da in das:
            for method in methods:
                avg_value = sum(res_dict[(method, da)][dom])/len(res_dict[(method, da)][dom])
                print('{:<25} {}\t{:.4f}'.format(method, da, avg_value))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'analyze results for exp0')
    parser.add_argument('--config_dir', type=str, required=True, help='config directory')
    args = parser.parse_args()
    
    run(args.config_dir)