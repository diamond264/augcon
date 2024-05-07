import os
from glob import glob

DIR = "/mnt/sting/hjyoon/projects/aaa/configs/supervised_adaptation/main_eval"
baseline_results = glob(
    os.path.join(
        DIR,
        "baseline",
        "ichar",
        "metasimclr",
        "finetune",
        "10shot",
        "linear",
        "*",
        "*.yaml.log",
    )
)
sup_results = glob(
    os.path.join(
        DIR,
        "sup",
        "ichar",
        "metasimclr",
        "finetune",
        "10shot",
        "linear",
        "*",
        "*.yaml.log",
    )
)

br = []
for b in baseline_results:
    with open(b, "r") as f:
        content = f.readlines()
        try:
            br.append(float(content[-1].split("F1: ")[1].split(",")[0]))
        except:
            br.append(0.0)

sr = []
for s in sup_results:
    with open(s, "r") as f:
        content = f.readlines()
        try:
            sr.append(float(content[-1].split("F1: ")[1].split(",")[0]))
        except:
            sr.append(0.0)

print(f"Baseline: {sum(br)/len(br)}")
print(f"Supervised: {sum(sr)/len(sr)}")
