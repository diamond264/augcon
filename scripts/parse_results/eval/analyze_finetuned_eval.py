import os
import argparse
from glob import glob
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/mnt/sting/hjyoon/projects/aaa/configs/imwut/main_eval')
    parser.add_argument('--metric', type=str, required=False, default='f1')
    parser.add_argument('--print_domains', action='store_true')
    parser.add_argument('--sort', action='store_true')
    parser.add_argument('--pretext', nargs='+', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--datasets', nargs='+', type=str, default=['ichar', 'hhar', 'pamap2', 'dsa'])
    parser.add_argument('--shots', nargs='+', type=int, default=[10, 1, 2, 5])
    parser.add_argument('--setting', type=str, default='linear')
    args = parser.parse_args()
    return args

SORT_LIST_ICHAR = [
    'WA0002-bkkim',
    'PH0007-jskim',
    'WA0003-hskim',
    'PH0012-thanh',
    'WA4697-jhryu',
    'PH0014-wjlee',
    'PH0034-ykha',
    'PH0038-iygoo',
    'PH0041-hmkim',
    'PH0045-sjlee'
]

SORT_LIST_HHAR = [
    'b-lgwatch',
    'd-lgwatch',
    'a-nexus4',
    'c-nexus4',
    'f-nexus4',
    'a-s3',
    'c-s3',
    'f-s3',
    'a-s3mini',
    'c-s3mini',
    'f-s3mini',
    'a-lgwatch',
    'c-lgwatch',
    'f-lgwatch',
    'b-nexus4',
    'd-nexus4',
    'b-s3',
    'd-s3',
    'b-s3mini',
    'd-s3mini'
]

SORT_LIST_PAMAP2 = [
    'wrist',
    'chest',
    'ankle'
]

SORT_LIST_DSA = [
    'T',
    'RA',
    'LA',
    'RL',
    'LL'
]

def run(args):
    for shot in args.shots:
        for dataset in args.datasets:
            res = defaultdict(list)
            for pretext in args.pretext:
                logs = glob(os.path.join(args.dir, dataset, pretext, 'finetune_target', f'{shot}shot', args.setting, f'seed{args.seed}', '*.log'))
                logs.sort()
                for log in logs:
                    domain = log.split('target_')[1].split('.yaml')[0]
                    # print(domain)
                    with open(log, 'r') as f:
                        content = f.read()
                        content = content.split('\n')[-2]
                        # print(content)
                        if args.metric == 'acc':
                            score = float(content.split('Acc(1): ')[1].split(', ')[0])
                        elif args.metric == 'f1':
                            score = float(content.split('F1: ')[1].split(', ')[0])
                        res[domain].append(str(score))
            
            args.sort = True
            if args.sort:
                if 'ichar' == dataset:
                    sort_list = SORT_LIST_ICHAR
                    sort_list = [f'domain_{d}' for d in sort_list]
                elif 'hhar' == dataset:
                    sort_list = SORT_LIST_HHAR
                    sort_list = [f'user_{d.split("-")[0]}_model_{d.split("-")[1]}' for d in sort_list]
                elif 'pamap2' == dataset:
                    sort_list = SORT_LIST_PAMAP2
                    sort_list = [f'domain_{d}' for d in sort_list]
                elif 'dsa' == dataset:
                    sort_list = SORT_LIST_DSA
                    sort_list = [f'domain_{d}' for d in sort_list]
                res_sorted = {}
                for d in sort_list:
                    res_sorted[d] = res[d]
                res = res_sorted
            
            sum = [0]*len(res[list(res.keys())[0]])
            for d, s in res.items():
                if not args.print_domains:
                    print(' '.join(s))
                else:
                    print(d, s)
                sum = [float(x) + float(y) for x, y in zip(sum, s)]
            print(' '.join([str(s/len(res)) for s in sum]))
    
if __name__ == '__main__':
    args = parse_args()
    run(args)