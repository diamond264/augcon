import os
import argparse
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretext', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    return args

def run(args):
    res = {}
    comp = {}
    CONFIG_PATH = f'/mnt/sting/hjyoon/projects/aaa/configs/imwut/main_hps/{args.dataset}/{args.pretext}/finetune/'

    folders = os.listdir(CONFIG_PATH)

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
    for d, s in max_values.items():
        print(d, s)


if __name__ == '__main__':
    args = parse_args()
    run(args)