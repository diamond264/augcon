import os
import argparse
from glob import glob

# /mnt/sting/hjyoon/projects/aaa/configs/imwut/main_eval/ichar/simclr/finetune/5shot/linear/seed0
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pretext', type=str, default='simclr')
    parser.add_argument('--dataset', type=str, default='i')
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--setting', type=str, default='endtoend')
    args = parser.parse_args()
    return args

domain_num = {
    'ichar' : 10,
    'hhar' : 20,
    'pamap2' : 3,
    'dsa' : 5
}

PRETEXTS = ["cpc", "simclr", "tpn", "autoencoder"]

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
    res = {"cpc" : {}, "simclr" : {}, "tpn" : {}, "autoencoder" : {}}

    if args.dataset.startswith('i'):
        dataset = 'ichar'
    elif args.dataset.startswith('h'):
        dataset = 'hhar'
    elif args.dataset.startswith('p'):
        dataset = 'pamap2'
    elif args.dataset.startswith('d'):
        dataset = 'dsa'

    for pretext in PRETEXTS:
        CONFIG_PATH = f'/mnt/sting/hjyoon/projects/aaa/configs/imwut/main_eval/{dataset}/{pretext}/finetune/{args.shot}shot/{args.setting}/seed{args.seed}'

        files = glob(os.path.join(CONFIG_PATH, '*.log'))
        # assert(len(files) == domain_num[args.dataset])

        for log in files:
            domain = log.split('target_')[1].split('.yaml')[0]
            # print(domain)
            with open(log, 'r') as f:
                content = f.read()
                content = content.split('\n')[-2]
                # print(content)
                try:
                    score = float(content.split('F1: ')[1].split(', ')[0])
                    res[pretext][domain] = score
                except:
                    res[pretext][domain] = -1

        if 'ichar' in CONFIG_PATH:
            sort_list = SORT_LIST_ICHAR
            sort_list = [f'domain_{d}' for d in sort_list]
        elif 'hhar' in CONFIG_PATH:
            sort_list = SORT_LIST_HHAR
            sort_list = [f'user_{d.split("-")[0]}_model_{d.split("-")[1]}' for d in sort_list]
        elif 'pamap2' in CONFIG_PATH:
            sort_list = SORT_LIST_PAMAP2
            sort_list = [f'domain_{d}' for d in sort_list]
        elif 'dsa' in CONFIG_PATH:
            sort_list = SORT_LIST_DSA
            sort_list = [f'domain_{d}' for d in sort_list]
        res_sorted = {}
        for d in sort_list:
            if d in res[pretext]:
                res_sorted[d] = res[pretext][d]
        res[pretext] = res_sorted

    print(f'{PRETEXTS[0]} {PRETEXTS[1]} {PRETEXTS[2]} {PRETEXTS[3]}')
    for d1, d2, d3, d4 in zip(res[PRETEXTS[0]].keys(), res[PRETEXTS[1]].keys(), res[PRETEXTS[2]].keys(),
                              res[PRETEXTS[3]].keys()):
        print(f'{d1} {d2} {d3} {d4}')
    print("------------------------------------------------")
    for s1, s2, s3, s4 in zip(res[PRETEXTS[0]].values(), res[PRETEXTS[1]].values(), res[PRETEXTS[2]].values(), res[PRETEXTS[3]].values()):
        print(f'{s1} {s2} {s3} {s4}')



if __name__ == '__main__':
    args = parse_args()
    run(args)