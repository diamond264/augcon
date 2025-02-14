import os
import numpy as np
import time
import wandb
import psutil

from pidstat import pidstat_cpu_mem
from procps import procps_cpu_mem
from top import top_cpu_mem


pid = os.getpid()
time = time.time()
day_hr_mn = time.strftime("%d_%H-%M")

m_interval = 1

# Initialize dictionaries to store metrics
metrics = {
    'top': {'cpu': {}, 'mem': {}},
    'pidstat': {'cpu': {}, 'mem': {}},
    'psutil': {'cpu': {}, 'mem': {}},
    'procps': {'cpu': {}, 'mem': {}}
}

# Function to perform matrix multiplication and collect metrics
def perform_computation(size):
    arr = np.random.rand(size, size)
    arr = arr @ arr.T

    # top snapshot
    t_cpu, t_mem = top_cpu_mem(pid)
    metrics['top']['cpu'][size] = t_cpu
    metrics['top']['mem'][size] = t_mem

    # pidstat snapshot
    cpu, mem = pidstat_cpu_mem(pid, interval=m_interval)
    metrics['pidstat']['cpu'][size] = float(cpu)
    metrics['pidstat']['mem'][size] = float(mem)

    # psutil
    process = psutil.Process(pid)
    cpu = process.cpu_percent(interval=m_interval)
    mem = process.memory_percent()
    metrics['psutil']['cpu'][size] = float(cpu)
    metrics['psutil']['mem'][size] = float(mem)

    # procps
    cpu, mem = procps_cpu_mem(pid)
    metrics['procps']['cpu'][size] = cpu
    metrics['procps']['mem'][size] = mem

# Run the experiment
for size in range(1000, 10000, 2000):
    perform_computation(size)
    print(f"Completed computation for matrix size: {size}")

# Log data to W&B after the experiment
for method in metrics:
    run = wandb.init(project="sensys", job_type=method, name=f"{method}_{day_hr_mn}", reinit=True)
    for size in metrics[method]['cpu']:
        wandb.log({"matrix_size": size, "cpu": metrics[method]['cpu'][size], "mem": metrics[method]['mem'][size]})
    run.finish()
