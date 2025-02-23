from glob import glob
from fire import Fire


def parse_results():
    result_files = glob("./overhead_output/*.txt")
    for file in result_files:
        lines = []
        with open(file) as f:
            lines = f.readlines()
        if len(file.split("/")[-1].split("_")) < 2:
            continue
        device = file.split("/")[-1].split("_")[0]
        method = file.split("/")[-1].split("_")[1]
        if device == "test":  # Skip test files
            continue
        print(f"Device: {device}, Method: {method}")

        idle = {"CPU": [], "RAM": []}
        adapt = {"CPU": [], "RAM": []}
        adapt_time = -1
        tune = {"CPU": [], "RAM": []}
        tune_time = -1
        for line in lines:
            if "Idle" in line:
                if "CPU" in line:
                    cpu_usage = float(
                        line.split("CPU Usage: ")[1].split("%")[0].strip()
                    )
                    idle["CPU"].append(cpu_usage)
                if "RAM" in line:
                    ram_usage = float(
                        line.split("RAM Usage: ")[1].split("MB")[0].strip()
                    )
                    idle["RAM"].append(ram_usage)
            if "Adapt" in line:
                if "CPU" in line:
                    cpu_usage = float(
                        line.split("CPU Usage: ")[1].split("%")[0].strip()
                    )
                    adapt["CPU"].append(cpu_usage)
                if "RAM" in line:
                    ram_usage = float(
                        line.split("RAM Usage: ")[1].split("MB")[0].strip()
                    )
                    adapt["RAM"].append(ram_usage)
                if "Time" in line:
                    adapt_time = float(line.split("Time: ")[1].strip())
            if "tune" in line:
                if "CPU" in line:
                    cpu_usage = float(
                        line.split("CPU Usage: ")[1].split("%")[0].strip()
                    )
                    tune["CPU"].append(cpu_usage)
                if "RAM" in line:
                    ram_usage = float(
                        line.split("RAM Usage: ")[1].split("MB")[0].strip()
                    )
                    tune["RAM"].append(ram_usage)
                if "Time" in line:
                    tune_time = float(line.split("Time: ")[1].strip())
        idle_ram = sum(idle["RAM"]) / len(idle["RAM"])
        idle_cpu = sum(idle["CPU"]) / len(idle["CPU"])
        adapt_ram = sum(adapt["RAM"]) / len(adapt["RAM"])
        adapt_cpu = sum(adapt["CPU"]) / len(adapt["CPU"])
        tune_ram = sum(tune["RAM"]) / len(tune["RAM"])
        tune_cpu = sum(tune["CPU"]) / len(tune["CPU"])
        adapt_cpu = adapt_cpu - idle_cpu
        adapt_ram = adapt_ram - idle_ram
        tune_cpu = tune_cpu - idle_cpu
        tune_ram = tune_ram - idle_ram
        print("Domain Adaptation:")
        print(f"CPU Usage: {adapt_cpu}%")
        print(f"RAM Usage: {adapt_ram}MB")
        print(f"Time: {adapt_time}s")
        print("")
        print("Fine-tuning:")
        print(f"CPU Usage: {tune_cpu}%")
        print(f"RAM Usage: {tune_ram}MB")
        print(f"Time: {tune_time}s")
        print("")
    pass


if __name__ == "__main__":
    Fire(parse_results)
