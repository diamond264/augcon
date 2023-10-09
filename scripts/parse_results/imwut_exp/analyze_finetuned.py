import os
import argparse
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('--metric', type=str, required=False, default='f1')
    parser.add_argument('--print_scores', action='store_true')
    args = parser.parse_args()
    return args

def run(args):
    res = {}
    logs = glob(os.path.join(args.dir, '*.log'))
    logs.sort()
    for log in logs:
        domain = log.split('target_')[1].split('.yaml')[0]
        # print(domain)
        with open(log, 'r') as f:
            content = f.read()
            content = content.split('\n')[-2]
            if args.metric == 'acc':
                score = float(content.split('Acc(1): ')[1].split(', ')[0])
            elif args.metric == 'f1':
                score = float(content.split('F1: ')[1].split(', ')[0])
            res[domain] = score
    
    for d, s in res.items():
        if args.print_scores:
            print(s)
        else:
            print(d, s)
    
if __name__ == '__main__':
    args = parse_args()
    run(args)