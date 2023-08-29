import os
import argparse
from glob import glob
from collections import defaultdict

def run(config_dir):
    ### OPTIONS
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    loss_res = {}
    knn_acc_res = {}
    acc_res = {}
    for domain in domains:
        loss_res[domain] = {"in": 0, "out": 0}
        knn_acc_res[domain] = {"in": 0, "out": 0}
        acc_res[domain] = {"in": 0, "out": 0}
        w_config = glob(os.path.join(config_dir, f'pt_w_{domain}', f'SimSiam_DomainNet_ft_{domain}.yaml.log'))[0]
        wo_config = glob(os.path.join(config_dir, f'pt_wo_{domain}', f'SimSiam_DomainNet_ft_{domain}.yaml.log'))[0]
        
        with open(w_config, 'r') as f:
            content = f.read()
            if content == "":
                print(f'no file {w_config}')
                continue
            
            if 'Pre-trained task loss:' not in content:
                print(f'no pre-trained task loss in {w_config}')
                continue
            
            if 'KNN Acc: ' not in content:
                print(f'no knn acc in {w_config}')
                continue
            
            if 'Validation Loss: ' not in content:
                print(f'no validation loss in {config}')
                continue
            
            loss = float(content.split('Pre-trained task loss: ')[-1].split('\tStd:')[0])
            knn_acc = float(content.split('KNN Acc: ')[-1].split('%')[0])
            acc = float(content.split('Acc(1): ')[-1].split(', Acc(5): ')[0])
            loss_res[domain]["in"] = loss
            knn_acc_res[domain]["in"] = knn_acc
            acc_res[domain]["in"] = acc
        
        with open(wo_config, 'r') as f:
            content = f.read()
            if content == "":
                print(f'no file {wo_config}')
                continue
            
            if 'Pre-trained task loss:' not in content:
                print(f'no pre-trained task loss in {wo_config}')
                continue
            
            if 'KNN Acc: ' not in content:
                print(f'no knn acc in {wo_config}')
                continue
            
            if 'Validation Loss: ' not in content:
                print(f'no validation loss in {wo_config}')
                continue
            
            loss = float(content.split('Pre-trained task loss: ')[-1].split('\tStd:')[0])
            knn_acc = float(content.split('KNN Acc: ')[-1].split('%')[0])
            acc = float(content.split('Acc(1): ')[-1].split(', Acc(5): ')[0])
            f1 = float(content.split('F1: ')[-1].split(',')[0])
            loss_res[domain]["out"] = loss
            knn_acc_res[domain]["out"] = knn_acc
            acc_res[domain]["out"] = acc
    
    ## pretty print results as table
    # columns
    print(f'\t\t', end='')
    for domain in domains:
        print(f'{domain[:4]}\t', end='')
    print()
    
    print(f'loss(in)\t', end='')
    for domain in domains:
        print(f'{loss_res[domain]["in"]:.4f}\t', end='')
    print()
    
    print(f'loss(out)\t', end='')
    for domain in domains:
        print(f'{loss_res[domain]["out"]:.4f}\t', end='')
    print()
    print()
    
    print(f'knn(in)\t\t', end='')
    for domain in domains:
        print(f'{knn_acc_res[domain]["in"]:.2f}\t', end='')
    print()
    
    print(f'knn(out)\t', end='')
    for domain in domains:
        print(f'{knn_acc_res[domain]["out"]:.2f}\t', end='')
    print()
    print()
    
    print(f'acc(in)\t\t', end='')
    for domain in domains:
        print(f'{acc_res[domain]["in"]:.2f}\t', end='')
    print()
    
    print(f'acc(out)\t', end='')
    for domain in domains:
        print(f'{acc_res[domain]["out"]:.2f}\t', end='')
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'analyze results for exp1')
    parser.add_argument('--config_dir', type=str, required=True, help='config directory')
    args = parser.parse_args()
    
    run(args.config_dir)