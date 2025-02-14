import os
import subprocess

def procps_cpu_mem(pid):
    """
    Returns (cpu_percent, mem_percent) for the given pid by parsing `ps`.
    This is a snapshot, not an average over time.
    """
    # Example fields: -o %cpu,%mem=only show CPU and MEM columns
    cmd = f"ps -p {pid} -o %cpu,%mem --no-headers"
    output = subprocess.check_output(cmd, shell=True, text=True).strip()
    if not output:
        return 0.0, 0.0
    
    cpu_str, mem_str = output.split()
    return float(cpu_str), float(mem_str)

def procps_all():
    """
    Returns all processes' CPU and MEM usage by parsing `ps`.
    This is a snapshot, not an average over time.
    """
    cmd = "ps -e -o pid,%cpu,%mem --no-headers"
    output = subprocess.check_output(cmd, shell=True, text=True).strip()
    if not output:
        return {}
    
    processes = {}
    for line in output.splitlines():
        line = line.strip()
        cols = line.split()
        pid = int(cols[0])
        cpu = float(cols[1])
        mem = float(cols[2])
        processes[pid] = (cpu, mem)

    total_cpu = sum(cpu for cpu, _ in processes.values())
    total_mem = sum(mem for _, mem in processes.values())
    
    return total_cpu, total_mem
