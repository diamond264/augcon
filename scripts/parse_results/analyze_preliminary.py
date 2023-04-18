import os
from glob import glob

log_dir = '/mnt/sting/hjyoon/projects/augcontrast/logs'
logs = glob(os.path.join(log_dir, '*.log'))

shots = ['1', '2', '5', '10']
users = ['a', 'b', 'c', 'd', 'e', 'f']
models = ['nexus4', 's3', 's3mini', 'lgwatch']

for shot in shots:
    print(f'[{shot} shot]')
    accs = {}
    accs2 = {}
    for u in users:
        accs[u] = {
            'in_domain': [],
            'out_domain': []
        }
        accs2[u] = {}
        
    for except_u in users:
        for u in users:
            fname = f'pretrained_except_{except_u}_finetuned_{u}_shot_{shot}.log'
            log_path = os.path.join(log_dir, fname)
            
            log_content = ""
            with open(log_path, 'r') as f:
                log_content = f.read()
                
            final_acc = log_content.split("Acc@1 ")[-1].split(" Acc@5")[0]
            try: final_acc = float(final_acc)
            except:
                print(f'{fname} has problem')
                continue
            
            if except_u == u:
                accs[u]['out_domain'].append(final_acc)
            else:
                accs[u]['in_domain'].append(final_acc)
            accs2[u][except_u] = final_acc

    in_mean_vals = []
    out_mean_vals = []
    for u in users:
        if len(accs[u]['out_domain']) == 0: continue
        in_mean_val = sum(accs[u]['in_domain'])/len(accs[u]['in_domain'])
        out_mean_val = sum(accs[u]['out_domain'])/len(accs[u]['out_domain'])
        print(f'[user {u}] - in-domain: {round(in_mean_val, 2)} / out-domain: {round(out_mean_val, 2)}')
        in_mean_vals.append(in_mean_val)
        out_mean_vals.append(out_mean_val)
    print(f'in-domain avg: {round(sum(in_mean_vals)/len(in_mean_vals), 2)} / out-domain avg: {round(sum(out_mean_vals)/len(out_mean_vals), 2)}')
    print('')

    for m in models:
        accs[m] = {
            'in_domain': [],
            'out_domain': []
        }
        accs2[m] = {}

    for except_m in models:
        for m in models:
            fname = f'pretrained_except_{except_m}_finetuned_{m}_shot_{shot}.log'
            log_path = os.path.join(log_dir, fname)
            
            log_content = ""
            with open(log_path, 'r') as f:
                log_content = f.read()
                
            final_acc = log_content.split("Acc@1 ")[-1].split(" Acc@5")[0]
            try: final_acc = float(final_acc)
            except:
                print(f'{fname} has problem')
                continue
            
            if except_m == m:
                accs[m]['out_domain'].append(final_acc)
            else:
                accs[m]['in_domain'].append(final_acc)
            accs2[m][except_m] = final_acc

    in_mean_vals = []
    out_mean_vals = []
    for m in models:
        if len(accs[m]['out_domain']) == 0: continue
        in_mean_val = sum(accs[m]['in_domain'])/len(accs[m]['in_domain'])
        out_mean_val = sum(accs[m]['out_domain'])/len(accs[m]['out_domain'])
        print(f'[model {m}] - in-domain: {round(in_mean_val, 2)} / out-domain: {round(out_mean_val, 2)}')
        in_mean_vals.append(in_mean_val)
        out_mean_vals.append(out_mean_val)
    print(f'in-domain avg: {round(sum(in_mean_vals)/len(in_mean_vals), 2)} / out-domain avg: {round(sum(out_mean_vals)/len(out_mean_vals), 2)}')
    print('')

    print(accs2)
    print('')