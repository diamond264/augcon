from glob import glob
import fire


def run(path):
    log_files = glob(f"{path}/*.log")
    f1scores = []
    for log_file in log_files:
        with open(log_file, "r", encoding="utf-8") as f:
            last_line = f.read().strip().split("\n")[-1]
            f1score = float(last_line.strip().split("F1: ")[1].strip().split(",")[0])
            f1scores.append(float(f1score))
    avg_f1score = sum(f1scores) / len(f1scores)
    print(f"Average F1 score: {avg_f1score}")


if __name__ == "__main__":
    fire.Fire(run)
