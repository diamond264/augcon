import os
import argparse
from glob import glob

SORT_LIST_ICHAR = [
    'WA0002-bkkim',
    'PH0012-thanh',
    'PH0038-iygoo',
]

SORT_LIST_HHAR = [
    'a-nexus4',
    'c-s3',
    'f-lgwatch',
]
SORT_LIST_PAMAP2 = [
    'chest',
]

SORT_LIST_DSA = [
    'LA',
]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretext', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--setting', type=str)
    args = parser.parse_args()
    return args

def run(args):
    res = {}
    comp = {}
    if args.dataset.startswith('i'):
        dataset = 'ichar'
    elif args.dataset.startswith('h'):
        dataset = 'hhar'
    elif args.dataset.startswith('p'):
        dataset = 'pamap2'
    elif args.dataset.startswith('d'):
        dataset = 'dsa'
    CONFIG_PATH = f'/mnt/sting/hjyoon/projects/aaa/configs/imwut/main_hps/{dataset}/{args.pretext}/finetune/'

    folders = os.listdir(CONFIG_PATH)
    print(folders)

    for folder in folders:
        res[folder] = {}
        logs = glob(os.path.join(CONFIG_PATH, folder, '*.log'))
        if len(logs) != 0:
            sum = 0
            for log in logs:
                domain = log.split('target_')[1].split('.yaml')[0]
                with open(log, 'r') as f:
                    content = f.read()
                    content = content.split('\n')[-2]
                    score = float(content.split('F1: ')[1].split(', ')[0])
                    sum += score
                    res[folder][domain] = score
            comp[folder] = sum / len(logs)

    max_key = max(comp, key=comp.get)
    max_value = comp[max_key]
    max_values = res[max_key]

    print(f"Result for {args.pretext} in the dataset {args.dataset}")
    print(f"Best config : {max_key}, with the highest mean F1 score {max_value}")

    if dataset == 'ichar':
        sort_list = SORT_LIST_ICHAR
        sort_list = [f'domain_{d}' for d in sort_list]
    elif dataset == 'hhar':
        sort_list = SORT_LIST_HHAR
        sort_list = [f'user_{d.split("-")[0]}_model_{d.split("-")[1]}' for d in sort_list]
    elif dataset == 'pamap2':
        sort_list = SORT_LIST_PAMAP2
        sort_list = [f'domain_{d}' for d in sort_list]
    elif dataset == 'dsa':
        sort_list = SORT_LIST_DSA
        sort_list = [f'domain_{d}' for d in sort_list]

    for key in sort_list:
        if key in max_values:
            print(key, max_values[key])


if __name__ == '__main__':
    args = parse_args()
    run(args)
    # /mnt/sting/hjyoon/projects/aaa/configs/imwut/main_hps_finetune/pamap2/metatpn/finetune/lr0.1_ep0999/gpu6_target_domain_chest.yaml
    #  python scripts/parse_results/imwut_exp/analyze_finetuned_hps_meta.py --pretext metaautoencoder --dataset h