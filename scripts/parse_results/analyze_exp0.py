import os
from collections import defaultdict

def run():
    exp0_log_dir = '/home/hjyoon/projects/augcon/config/exp0/'
    
    ### OPTIONS
    das = [True, False]
    domains = [
        'PH0007-jskim',
        'PH0014-wjlee',
        'PH0038-iygoo',
        'PH0045-sjlee',
        'WA0003-hskim',
        'PH0012-thanh',
        'PH0034-ykha',
        'PH0041-hmkim',
        'WA0002-bkkim',
        'WA4697-jhryu'
    ]
    methods = ['cpc_random', 'metacpc_random', 'cpc_perdomain', 'metacpc_perdomain']
    
    res_dict = {}
    for da in das:
        for method in methods:
            res_dict[(method, da)] = {
                'in_domain': [],
                'out_domain': []
            }
            for src_dom in domains:
                for tgt_dom in domains:
                    finetune_dir = 'finetune_da_true' if da else 'finetune_da_false'
                    log_file = os.path.join(exp0_log_dir, 
                                            finetune_dir, 
                                            f'without_target_domain_{src_dom}',
                                            f'{method}_8421',
                                            f'tgt_target_domain_{tgt_dom}_seed0.yaml.log')
                    log_content = ""
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        if log_content == "":
                            print(f'no file {log_file}')
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
                print('{:<20} da_{}\t{:.4f}'.format(method, da, avg_value))

if __name__ == '__main__':
    run()