import os
import argparse
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    parser.add_argument("--metric", type=str, required=False, default="f1")
    parser.add_argument("--print_scores", action="store_true")
    parser.add_argument("--sort", action="store_true")
    args = parser.parse_args()
    return args


SORT_LIST_ICHAR = [
    "WA0002-bkkim",
    "PH0007-jskim",
    "WA0003-hskim",
    "PH0012-thanh",
    "WA4697-jhryu",
    "PH0014-wjlee",
    "PH0034-ykha",
    "PH0038-iygoo",
    "PH0041-hmkim",
    "PH0045-sjlee",
]

SORT_LIST_HHAR = [
    "b-lgwatch",
    "d-lgwatch",
    "a-nexus4",
    "c-nexus4",
    "f-nexus4",
    "a-s3",
    "c-s3",
    "f-s3",
    "a-s3mini",
    "c-s3mini",
    "f-s3mini",
    "a-lgwatch",
    "c-lgwatch",
    "f-lgwatch",
    "b-nexus4",
    "d-nexus4",
    "b-s3",
    "d-s3",
    "b-s3mini",
    "d-s3mini",
]

SORT_LIST_PAMAP2 = ["wrist", "chest", "ankle"]

SORT_LIST_DSA = ["T", "RA", "LA", "RL", "LL"]


def run(args):
    res = {}
    logs = glob(os.path.join(args.dir + "*", "*.log"))
    logs.sort()
    for log in logs:
        domain = log.split("target_")[1].split(".yaml")[0]
        # print(domain)
        with open(log, "r") as f:
            content = f.read()
            content = content.split("\n")[-2]
            # print(content)
            if args.metric == "acc":
                score = float(content.split("Acc(1): ")[1].split(", ")[0])
            elif args.metric == "f1":
                score = float(content.split("F1: ")[1].split(", ")[0])
            res[domain] = score

    if args.sort:
        if "ichar" in args.dir:
            sort_list = SORT_LIST_ICHAR
            sort_list = [f"domain_{d}" for d in sort_list]
        elif "hhar" in args.dir:
            sort_list = SORT_LIST_HHAR
            sort_list = [
                f'user_{d.split("-")[0]}_model_{d.split("-")[1]}' for d in sort_list
            ]
        elif "pamap2" in args.dir:
            sort_list = SORT_LIST_PAMAP2
            sort_list = [f"domain_{d}" for d in sort_list]
        elif "dsa" in args.dir:
            sort_list = SORT_LIST_DSA
            sort_list = [f"domain_{d}" for d in sort_list]
        res_sorted = {}
        for d in sort_list:
            res_sorted[d] = res[d]
        res = res_sorted

    for d, s in res.items():
        if args.print_scores:
            print(s)
        else:
            print(d, s)

    print("Mean:", sum(res.values()) / len(res))


if __name__ == "__main__":
    args = parse_args()
    run(args)
