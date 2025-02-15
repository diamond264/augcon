import os
from fire import Fire


def analyze(path):
    methods = ["simclr", "metasimclr", "cpc", "metacpc", "tpn", "metatpn"]
    for method in methods:
        method_log = os.path.join(path, f"{method}_overhead.log")
        lines = []
        with open(method_log, "r", encoding="utf-8") as f:
            lines = f.readlines()
        epoch_times = []
        gpu_vrams = []
        gpu_usages = []
        task_gen_times = []
        for line in lines:
            if line.startswith("Epoch time:"):
                epoch_time = float(line.split(":")[-1])
                epoch_times.append(epoch_time)
            if line.startswith("GPU Memory:"):
                gpu_vram = float(line.split(":")[-1])
                gpu_vrams.append(gpu_vram)
            if line.startswith("GPU Utilization:"):
                gpu_usage = float(line.split(":")[-1])
                gpu_usages.append(gpu_usage)
            if line.startswith("Task generation time:"):
                task_gen_time = float(line.split(":")[-1])
                task_gen_times.append(task_gen_time)
        print(f"Method: {method}")
        print(f"Average epoch time: {sum(epoch_times) / len(epoch_times)}")
        print(f"Average GPU VRAM: {sum(gpu_vrams) / len(gpu_vrams)}")
        print(f"Average GPU Usage: {sum(gpu_usages) / len(gpu_usages)}")
        if len(task_gen_times) > 0:
            print(
                f"Average Task generation time: {sum(task_gen_times) / len(task_gen_times)}"
            )
        print("\n")


if __name__ == "__main__":
    Fire(analyze)
