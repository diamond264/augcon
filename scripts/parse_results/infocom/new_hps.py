from fire import Fire
from glob import glob


def run(path):
    res = {}
    subdirs = glob(f"{path}/*")
    for subdir in subdirs:
        log_path = glob(f"{subdir}/*.log")[0]
        loss = 9999
        with open(log_path, "r", encoding="utf-8") as f:
            last_line = f.read().strip().split("\n")[-1]
            loss = float(last_line.strip().split("Loss: ")[1].strip().split(",")[0])
        setting = subdir.split("/")[-1]
        res[setting] = loss

    # the lower the better
    sorted_res = sorted(res.items(), key=lambda x: x[1])
    for setting, loss in sorted_res:
        print(f"{setting}: {loss}")
    print("Best setting:")
    print(sorted_res[0])


if __name__ == "__main__":
    Fire(run)
