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

if __name__ == "__main__":
    pid = os.getpid()
    cpu, mem = procps_cpu_mem(pid)
    print(f"Process {pid} usage => CPU: {cpu}%, MEM: {mem}%")
