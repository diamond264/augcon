import os
from glob import glob

log_dir = '/mnt/sting/hjyoon/projects/augcontrast/logs'
logs = glob(os.path.join(log_dir, '*.log'))

shots = ['1', '2', '5', '10']
users = ['a', 'b', 'c', 'd', 'f']
models = ['nexus4', 's3', 's3mini', 'lgwatch']

for shot in shots:
    print(f'[{shot} shot]')
    accs = {}
    accs2 = {}
    for u in users:
        for m in models:
            accs[(u, m)] = {
                'in_domain': [],
                'out_domain': []
            }
            accs2[(u, m)] = {}
        
    for except_u in users:
        for except_m in models:
            for u in users:
                for m in models:
                    if except_u==u and except_m!=m: continue
                    if except_u!=u and except_m==m: continue
                    fname = f'230418_pt_except_shot_{shot}_user_{except_u}_model_{except_m}_finetune_user_{u}_model_{m}.log'
                    log_path = os.path.join(log_dir, fname)
                    
                    log_content = ""
                    try:
                        with open(log_path, 'r') as f:
                            log_content = f.read()
                    except:
                        print(f'no file {log_path}')
                        continue
                        
                    final_acc = log_content.split("Acc@1 ")[-1].split(" Acc@5")[0]
                    try: final_acc = float(final_acc)
                    except:
                        print(f'{fname} has problem')
                        continue
                    
                    if except_u == u and except_m == m:
                        accs[(u, m)]['out_domain'].append(final_acc)
                    else:
                        accs[(u, m)]['in_domain'].append(final_acc)
                    accs2[(u, m)][(except_u, except_m)] = final_acc

    in_mean_vals = []
    out_mean_vals = []
    for u in users:
        for m in models:
            if len(accs[(u, m)]['out_domain']) == 0: continue
            in_mean_val = sum(accs[(u, m)]['in_domain'])/len(accs[(u, m)]['in_domain'])
            out_mean_val = sum(accs[(u, m)]['out_domain'])/len(accs[(u, m)]['out_domain'])
            print(f'[user {u}, model {m}] - in-domain: {round(in_mean_val, 2)} / out-domain: {round(out_mean_val, 2)}')
            in_mean_vals.append(in_mean_val)
            out_mean_vals.append(out_mean_val)
    print(f'in-domain avg: {round(sum(in_mean_vals)/len(in_mean_vals), 2)} / out-domain avg: {round(sum(out_mean_vals)/len(out_mean_vals), 2)}')
    print('')

    print(accs2)
    print('')